"""Human study on real low-quality ads: pairwise comparisons (2AFC) of methods, served from this machine to any browser.

People judge the methods on the real test segments (adup.real_ads.human_eval_set), where no ground truth exists. Each
trial shows the same segment restored by two conditions (order and side random, names hidden) and asks
  1. which looks better overall (forced choice)
  2. whether text, prices or logos are unreadable or changed on either side
  3. whether faces look unnatural on either side
Conditions are the methods' outputs from adup.bench.run / training/dove/infer.py, plus for Meta ads the same frames of
Meta's own 720p rendition upscaled with bicubic ("meta_hd"), a real higher-quality anchor. The server hands every rater
the pairs with the fewest votes that they have not judged, so votes spread evenly; it runs on the local network, so
raters can use their phones (a phone shows one side at a time with an A / B switch, time-synced; a wide screen shows
both side by side).

  build    collect the conditions of every segment into a study folder (videos remuxed for streaming, study.json, page)
  serve    serve the study; votes are appended to <study>/votes.csv
  analyze  Bradley-Terry strengths with bootstrap intervals, win rates, text / face issue rates, rater agreement

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.bench.human_study build --segments data/eval_sets/real_ugc_v1 \\
      --runs outputs/bench/real_ugc_v1 --methods bicubic realesrgan dove --anchor --out outputs/human_study/real_ugc_v1
  .venv-iqa/bin/python -m adup.bench.human_study serve outputs/human_study/real_ugc_v1 [--port 8765]
  .venv-iqa/bin/python -m adup.bench.human_study analyze outputs/human_study/real_ugc_v1
Adding a method later: rerun build with it; existing votes stay valid (pair ids are "<segment>|<a>|<b>").
"""

import argparse
import csv
import datetime
import hashlib
import http.server
import itertools
import json
import os
import random
import secrets
import socket
import subprocess
import threading
import urllib.parse

import numpy as np
import pandas as pd

from adup.media import ENC, FF, probe

def blind_name(segment, cond):
    """Video file name that does not reveal the method (raters can see URLs, e.g. by long-pressing a video)."""
    return hashlib.sha1(f"{segment}|{cond}|adup-human-study".encode()).hexdigest()[:16] + ".mp4"


VOTE_FIELDS = ["time", "rater", "pair", "segment", "left", "right", "choice", "winner", "text", "face", "watch_s", "mode"]
PAGE = os.path.join(os.path.dirname(__file__), "human_study.html")


# ---------------------------------------------------------------- build
def build(args):
    seg = pd.read_csv(os.path.join(args.segments, "segments.csv"), dtype={"ad_id": str})
    vdir = os.path.join(args.out, "videos")
    os.makedirs(vdir, exist_ok=True)
    conds = list(args.methods) + (["meta_hd"] if args.anchor else [])
    segments, pairs = [], []
    for r in seg.itertuples():
        have = []
        for c in conds:
            dst = os.path.join(vdir, blind_name(r.segment, c))
            if c == "meta_hd":
                src = os.path.join(args.segments, "anchors_hd", f"{r.segment}.mp4")
                if not os.path.exists(src):
                    continue
                if not os.path.exists(dst):              # Meta's 720p, upscaled like a player would, to the output size
                    w, h = probe(src)[:2]
                    s = args.out_short / min(w, h)
                    subprocess.run([*FF, "-y", "-i", src, "-vf", f"scale={round(w * s / 2) * 2}:{round(h * s / 2) * 2}:flags=bicubic",
                                    "-c:v", "libx264", "-preset", "medium", "-crf", "10", *ENC, "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", "-an", dst], check=True)
            else:
                src = os.path.join(args.runs, c, f"{r.segment}.mp4")
                if not os.path.exists(src):
                    print(f"missing {c} for {r.segment}", flush=True)
                    continue
                if not os.path.exists(dst):              # same stream, moov atom first so browsers start at once
                    subprocess.run([*FF, "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", "-an", dst], check=True)
            have.append(c)
        w, h, fps, n = probe(os.path.join(vdir, blind_name(r.segment, have[0]))) if have else (0, 0, 0, 0)
        segments.append({"id": r.segment, "platform": r.platform, "industry": r.industry, "input": [int(r.width), int(r.height)],
                         "output": [w, h], "fps": round(fps, 3), "frames": n, "conditions": have})
        pairs += [f"{r.segment}|{a}|{b}" for a, b in itertools.combinations(sorted(have), 2)]
    study = {"name": os.path.basename(os.path.normpath(args.out)), "created": datetime.datetime.now().isoformat(timespec="seconds"),
             "conditions": conds, "segments": segments, "pairs": pairs, "votes_per_pair": args.votes_per_pair,
             "per_session": args.per_session}
    with open(os.path.join(args.out, "study.json"), "w") as f:
        json.dump(study, f, indent=1)
    with open(PAGE) as f, open(os.path.join(args.out, "index.html"), "w") as g:
        g.write(f.read())
    print(f"{len(segments)} segments, {len(pairs)} pairs, conditions {conds} -> {args.out}")


# ---------------------------------------------------------------- serve
class Study:
    def __init__(self, folder):
        self.folder = folder
        self.study = json.load(open(os.path.join(folder, "study.json")))
        self.votes_path = os.path.join(folder, "votes.csv")
        self.lock = threading.Lock()
        self.votes = pd.read_csv(self.votes_path).to_dict("records") if os.path.exists(self.votes_path) else []
        self.issued = {}                  # token -> (pair, left, right): the page never sees which method is which

    def next_pair(self, rater):
        with self.lock:
            done = {v["pair"] for v in self.votes if v["rater"] == rater}
            count = {}
            for v in self.votes:
                count[v["pair"]] = count.get(v["pair"], 0) + 1
            last = [v["segment"] for v in self.votes if v["rater"] == rater][-2:]
        todo = [p for p in self.study["pairs"] if p not in done]
        if not todo:
            return {"done": True, "judged": len(done)}
        least = min(count.get(p, 0) for p in todo)
        pool = [p for p in todo if count.get(p, 0) == least]
        fresh = [p for p in pool if p.split("|")[0] not in last]      # not the same ad twice in a row
        pair = random.choice(fresh or pool)
        segment, a, b = pair.split("|")
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        s = next(x for x in self.study["segments"] if x["id"] == segment)
        token = secrets.token_hex(8)
        with self.lock:
            self.issued[token] = (pair, left, right)
        return {"token": token, "judged": len(done), "total": len(self.study["pairs"]),
                "per_session": self.study["per_session"], "aspect": s["output"],
                "left": f"videos/{blind_name(segment, left)}", "right": f"videos/{blind_name(segment, right)}"}

    def add_vote(self, v):
        with self.lock:
            issued = self.issued.pop(v.get("token", ""), None)
        if issued is None:
            raise LookupError("unknown or expired trial")
        pair, left, right = issued
        segment = pair.split("|")[0]
        if v.get("choice") not in ("left", "right"):
            raise ValueError("bad vote")
        row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "rater": str(v["rater"])[:40],
               "pair": pair, "segment": segment, "left": left, "right": right, "choice": v["choice"],
               "winner": left if v["choice"] == "left" else right, "text": v.get("text", ""), "face": v.get("face", ""),
               "watch_s": round(float(v.get("watch_s", 0)), 1), "mode": v.get("mode", "")}
        with self.lock:
            new = not os.path.exists(self.votes_path)
            with open(self.votes_path, "a", newline="") as f:
                w = csv.DictWriter(f, VOTE_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            self.votes.append(row)
            return sum(1 for x in self.votes if x["rater"] == row["rater"])


def serve(args):
    st = Study(args.study)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=args.study, **k)

        def log_message(self, fmt, *a):
            if "/api/" in (a[0] if a else ""):
                super().log_message(fmt, *a)

        def reply(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/next":
                rater = urllib.parse.parse_qs(u.query).get("rater", [""])[0].strip()
                return self.reply(st.next_pair(rater) if rater else {"error": "no rater"}, 200 if rater else 400)
            if u.path not in ("/", "/index.html") and not u.path.startswith("/videos/"):
                return self.reply({"error": "forbidden"}, 403)          # study.json, votes and results stay private
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/vote":
                return self.reply({"error": "not found"}, 404)
            try:
                v = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                return self.reply({"ok": True, "judged": st.add_vote(v)})
            except LookupError as e:                                # e.g. the server restarted: the page fetches a new trial
                return self.reply({"error": str(e)}, 409)
            except (ValueError, KeyError) as e:
                return self.reply({"error": str(e)}, 400)

    srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    ips = {"localhost"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    print(f"study {st.study['name']}: {len(st.study['pairs'])} pairs, {len(st.votes)} votes so far")
    for ip in sorted(ips):
        print(f"  open http://{ip}:{args.port}/")
    print("votes -> " + st.votes_path + "   (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


# ---------------------------------------------------------------- analyze
def bradley_terry(conds, votes, iters=500):
    """Strengths (log scale, mean 0) from (winner, loser) pairs; MM updates with a tiny prior against separation."""
    k = {c: i for i, c in enumerate(conds)}
    wins = np.full((len(conds), len(conds)), 0.1 / max(len(conds) - 1, 1))
    np.fill_diagonal(wins, 0)
    for w, l in votes:
        wins[k[w], k[l]] += 1
    n = wins + wins.T
    p = np.ones(len(conds))
    for _ in range(iters):
        p = wins.sum(1) / (n / (p[:, None] + p[None, :])).sum(1)
        p /= np.exp(np.log(p).mean())
    return np.log(p)


def analyze(args):
    study = json.load(open(os.path.join(args.study, "study.json")))
    path = os.path.join(args.study, "votes.csv")
    if not os.path.exists(path):
        print("no votes yet")
        return
    v = pd.read_csv(path)
    seg = {s["id"]: s for s in study["segments"]}
    v["platform"] = v.segment.map(lambda s: seg[s]["platform"])
    v["loser"] = np.where(v.winner == v.left, v.right, v.left)
    conds = [c for c in study["conditions"] if c in set(v.left) | set(v.right)]
    rng = np.random.default_rng(0)
    out = [f"# 人评结果：{study['name']}", "",
           (f"{len(v)} 票，{v.rater.nunique()} 位评分人，{v.pair.nunique()} / {len(study['pairs'])} 组对比有票；"
            f"每票平均观看 {v.watch_s.mean():.0f} 秒。"), ""]
    res = {}
    for name, sub in [("全部", v)] + [(f"{p}", g) for p, g in v.groupby("platform")]:
        pairs = list(zip(sub.winner, sub.loser))
        cs = [c for c in conds if c in set(sub.left) | set(sub.right)]
        bt = bradley_terry(cs, pairs)
        boot = np.array([bradley_terry(cs, [pairs[i] for i in rng.integers(len(pairs), size=len(pairs))])
                         for _ in range(args.bootstrap)])
        ref = cs.index("bicubic") if "bicubic" in cs else 0
        lo, hi = np.percentile(boot - boot[:, [ref]], [2.5, 97.5], axis=0)     # difference to bicubic in each resample
        lo, hi = lo + bt[ref], hi + bt[ref]
        res[name] = {c: {"bt": float(bt[i] - bt[ref]), "lo": float(lo[i] - bt[ref]), "hi": float(hi[i] - bt[ref])} for i, c in enumerate(cs)}
        out += [f"## Bradley-Terry 强度（{name}，{len(sub)} 票；相对 bicubic，对数尺度，95% 自助区间）", "",
                "| 条件 | 强度 | 95% 区间 | 胜率 | 出场 |", "|---|---|---|---|---|"]
        for i in np.argsort(-bt):
            c = cs[i]
            appear = ((sub.left == c) | (sub.right == c)).sum()
            out.append(f"| {c} | {bt[i] - bt[ref]:+.2f} | [{lo[i] - bt[ref]:+.2f}, {hi[i] - bt[ref]:+.2f}] | "
                       f"{(sub.winner == c).sum() / max(appear, 1):.0%} | {appear} |")
        out.append("")
    out += ["## 两两胜率（行胜过列的比例，票数）", "", "| | " + " | ".join(conds) + " |", "|---" * (len(conds) + 1) + "|"]
    for a in conds:
        cells = []
        for b in conds:
            m = v[((v.left == a) & (v.right == b)) | ((v.left == b) & (v.right == a))]
            cells.append("—" if a == b or not len(m) else f"{(m.winner == a).mean():.0%} ({len(m)})")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    out += ["", "## 文字 / 人脸问题（被评分人标出的比例）", "", "| 条件 | 文字看不清或被改 | 人脸不自然 |", "|---|---|---|"]
    for c in conds:
        rows = v[(v.left == c) | (v.right == c)]
        t = rows[~rows.text.isin(["na"])]
        ts = np.where(t.left == c, "left", "right")
        f = rows[~rows.face.isin(["na"])]
        fs = np.where(f.left == c, "left", "right")
        tr = ((t.text == ts) | (t.text == "both")).mean() if len(t) else np.nan
        fr = ((f.face == fs) | (f.face == "both")).mean() if len(f) else np.nan
        out.append(f"| {c} | {tr:.0%}（{len(t)} 次有字） | {fr:.0%}（{len(f)} 次有人脸） |")
    multi = v.groupby("pair").filter(lambda g: len(g) >= 2)
    if len(multi):
        agree = multi.groupby("pair").winner.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean()
        out += ["", f"评分一致性：{multi.pair.nunique()} 组有 2 票以上，平均 {agree:.0%} 的票和该组多数意见一致。"]
    text = "\n".join(out)
    with open(os.path.join(args.study, "results.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.study, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--segments", required=True)
    b.add_argument("--runs", required=True)
    b.add_argument("--methods", nargs="+", required=True)
    b.add_argument("--anchor", action="store_true", help="add Meta's own 720p (bicubic to the output size) as a condition")
    b.add_argument("--out", required=True)
    b.add_argument("--out-short", type=int, default=1080)
    b.add_argument("--votes-per-pair", type=int, default=3, help="target votes per pair (shown to raters as progress)")
    b.add_argument("--per-session", type=int, default=40, help="suggested comparisons per rater session")
    s = sub.add_parser("serve")
    s.add_argument("study")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    a = sub.add_parser("analyze")
    a.add_argument("study")
    a.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    {"build": build, "serve": serve, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
