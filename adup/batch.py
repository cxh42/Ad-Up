"""Run a batch of make_pairs jobs unattended, and show its progress.

A batch file (configs/batches/<name>.yaml) lists tasks, each a make_pairs run over a specs file. `run` works through
them one at a time under adup.memguard: when available memory drops below min_free_gb the job is stopped, the runner
waits until resume_free_gb is free again and resumes (make_pairs skips finished ads). After a reboot, the same `run`
command continues where it stopped. The runner writes status.json and a self-refreshing status.html to the batch's
log directory every 30 s.

Usage (from the repo root):
  nohup .venv-iqa/bin/python -m adup.batch run configs/batches/v7_local.yaml > /dev/null 2>&1 &
  .venv-iqa/bin/python -m adup.batch status configs/batches/v7_local.yaml [--watch 30]
  xdg-open outputs/logs/batch_v7_local/status.html
"""

import argparse
import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import yaml

PY = sys.executable


def load(path):
    b = yaml.safe_load(open(path))
    b.setdefault("log_dir", f"outputs/logs/batch_{os.path.splitext(os.path.basename(path))[0]}")
    for t in b["tasks"]:
        t.setdefault("out", os.path.dirname(t["specs"]))
        t["grid"] = "--grid" in t["args"]
        t.setdefault("sec_per_ad", 70 if t["grid"] else 26)       # used until the task has finished a few ads
    os.makedirs(b["log_dir"], exist_ok=True)
    return b


def spec_ids(t):
    ids = [json.loads(line)["id"] for line in open(t["specs"])]
    return ids[:t["limit"]] if t.get("limit") else ids


def task_state(b, t):
    """Done / failed / remaining counts, seconds per ad and ETA of one task."""
    ids = spec_ids(t)
    metas = [os.path.join(t["out"], i, "meta.json") for i in ids]
    times = sorted(os.path.getmtime(m) for m in metas if os.path.exists(m))
    log = os.path.join(b["log_dir"], f"{t['name']}.log")
    failed = set()
    for f in (log + ".all", log):                             # earlier runs of the task are kept in .all
        if os.path.exists(f):
            failed |= {m.group(1) for m in re.finditer(r"\] (\S+) failed:", open(f, errors="replace").read())}
    failed -= {i for i, m in zip(ids, metas) if os.path.exists(m)}
    gaps = np.diff(times[-31:])
    gaps = gaps[gaps < 600]                                   # pauses and reboots are not part of the rate
    rate = float(np.median(gaps)) if len(gaps) >= 3 else t["sec_per_ad"]
    remaining = len(ids) - len(times) - len(failed)
    return {"name": t["name"], "total": len(ids), "done": len(times), "failed": len(failed), "remaining": remaining,
            "sec_per_ad": rate, "eta_s": remaining * rate, "last": times[-1] if times else None, "out": t["out"]}


def runner_state(b):
    p = os.path.join(b["log_dir"], "status.json")
    s = json.load(open(p)) if os.path.exists(p) else {}
    try:
        alive = bool(s.get("pid")) and any("adup.batch" in c for c in psutil.Process(s["pid"]).cmdline())
    except psutil.Error:
        alive = False
    if not alive and s.get("state") not in (None, "finished"):
        s["state"] = "stopped"
    return s


def fmt_t(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min"


def snapshot(b, path):
    tasks = [task_state(b, t) for t in b["tasks"]]
    r = runner_state(b)
    vm = psutil.virtual_memory()
    size = sum(os.path.getsize(os.path.join(d, f)) for t in b["tasks"] if os.path.isdir(t["out"])
               for d, _, fs in os.walk(t["out"]) for f in fs)
    return {"batch": os.path.basename(path), "tasks": tasks, "runner": r, "time": time.time(),
            "mem_avail_gb": vm.available / 2 ** 30, "mem_total_gb": vm.total / 2 ** 30,
            "disk_free_gb": shutil.disk_usage(".").free / 2 ** 30, "data_gb": size / 2 ** 30,
            "eta_s": sum(t["eta_s"] for t in tasks), "resume_cmd": f"nohup {os.path.relpath(PY)} -m adup.batch run {path} "
                                                               f"> /dev/null 2>&1 &"}


STATES = {"running": "运行中", "waiting_memory": "等待内存（可用内存不足，已暂停）", "finished": "全部完成",
          "stopped": "未在运行（重启或被中断，用下面的命令续跑）", None: "尚未开始"}


def text(s):
    r = s["runner"]
    lines = [f"批处理 {s['batch']}   状态：{STATES.get(r.get('state'), r.get('state'))}"
             + (f"   当前任务：{r['task']}" if r.get("task") and r.get("state") in ("running", "waiting_memory") else ""),
             f"{'任务':12s}{'完成':>12s}{'失败':>6s}{'每条':>8s}{'剩余时间':>14s}"]
    for t in s["tasks"]:
        pct = 100 * t["done"] / max(t["total"], 1)
        lines.append(f"{t['name']:12s}{t['done']:>6d}/{t['total']:<5d}{t['failed']:>6d}{t['sec_per_ad']:>7.0f}s"
                     f"{fmt_t(t['eta_s']):>14s}   {'#' * int(pct / 5):20s} {pct:4.0f}%")
    lines.append(f"总剩余约 {fmt_t(s['eta_s'])}（预计 {datetime.datetime.fromtimestamp(s['time'] + s['eta_s']):%m-%d %H:%M} 完成）；"
                 f"可用内存 {s['mem_avail_gb']:.1f} / {s['mem_total_gb']:.0f} GB；已生成 {s['data_gb']:.1f} GB；磁盘剩余 {s['disk_free_gb']:.0f} GB")
    if r.get("state") == "stopped":
        lines.append(f"续跑：{s['resume_cmd']}")
    for e in r.get("events", [])[-5:]:
        lines.append(f"  {e}")
    return "\n".join(lines)


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Batch Progress</title><style>
:root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc; --bar: #2f6fd6; --ok: #2e8b57; --bad: #c0392b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ecebe6; --muted: #9a9993; --line: #2e2e2b; --bar: #6aa0ff; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 system-ui, "Noto Sans CJK SC", sans-serif; margin: 0; }}
main {{ max-width: 860px; margin: 0 auto; padding: 32px 16px; }} h1 {{ font-size: 22px; margin: 0; }}
.muted {{ color: var(--muted); }} .state {{ font-weight: 600; }} .state.running {{ color: var(--ok); }}
.state.stopped, .state.waiting_memory {{ color: var(--bad); }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }} td, th {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; }}
td.n {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
.bar {{ background: var(--line); border-radius: 4px; height: 10px; min-width: 120px; }} .bar i {{ display: block; height: 10px; border-radius: 4px; background: var(--bar); }}
code {{ font-size: 13px; word-break: break-all; }} ul {{ padding-left: 18px; }}
</style></head><body><main>
<h1>数据生成进度</h1><p class="muted">{batch} · 每 30 秒自动刷新 · 更新于 {now}</p>
<p>状态：<span class="state {state_key}">{state}</span>{task}</p>
<table><tr><th>任务</th><th>完成</th><th>进度</th><th>失败</th><th>每条</th><th>剩余时间</th></tr>{rows}</table>
<p>总剩余约 <b>{eta}</b>，预计 <b>{finish}</b> 完成。可用内存 {mem:.1f} / {mem_total:.0f} GB；已生成 {data:.1f} GB；磁盘剩余 {disk:.0f} GB。</p>
{resume}<h3>最近事件</h3><ul>{events}</ul>
<p class="muted">终端查看：<code>.venv-iqa/bin/python -m adup.batch status {path} --watch 30</code></p>
</main></body></html>"""


def write_html(b, path, s):
    r = s["runner"]
    rows = "".join(f"<tr><td>{html.escape(t['name'])}</td><td class=n>{t['done']} / {t['total']}</td>"
                   f"<td><div class=bar><i style='width:{100 * t['done'] / max(t['total'], 1):.1f}%'></i></div></td>"
                   f"<td class=n>{t['failed']}</td><td class=n>{t['sec_per_ad']:.0f} s</td><td class=n>{fmt_t(t['eta_s'])}</td></tr>"
                   for t in s["tasks"])
    page = PAGE.format(
        batch=html.escape(s["batch"]), now=f"{datetime.datetime.fromtimestamp(s['time']):%m-%d %H:%M:%S}",
        state_key=r.get("state") or "", state=STATES.get(r.get("state"), r.get("state")),
        task=f"，当前任务 <b>{html.escape(r['task'])}</b>" if r.get("task") and r.get("state") in ("running", "waiting_memory") else "",
        rows=rows, eta=fmt_t(s["eta_s"]), finish=f"{datetime.datetime.fromtimestamp(s['time'] + s['eta_s']):%m-%d %H:%M}",
        mem=s["mem_avail_gb"], mem_total=s["mem_total_gb"], data=s["data_gb"], disk=s["disk_free_gb"],
        resume=f"<p>续跑命令：<code>{html.escape(s['resume_cmd'])}</code></p>" if r.get("state") == "stopped" else "",
        events="".join(f"<li>{html.escape(e)}</li>" for e in reversed(r.get("events", [])[-8:])), path=html.escape(path))
    tmp = os.path.join(b["log_dir"], "status.html.tmp")
    with open(tmp, "w") as f:
        f.write(page)
    os.replace(tmp, os.path.join(b["log_dir"], "status.html"))


def run(path):
    b = load(path)
    sp = os.path.join(b["log_dir"], "status.json")
    old = json.load(open(sp)) if os.path.exists(sp) else {}
    st = {"pid": os.getpid(), "state": "running", "task": None, "events": old.get("events", [])[-20:]}

    def save(event=None):
        if event:
            st["events"].append(f"{datetime.datetime.now():%m-%d %H:%M} {event}")
        with open(sp + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(sp + ".tmp", sp)
        write_html(b, path, snapshot(b, path))

    save("开始 / 续跑")
    for t in b["tasks"]:
        st["task"] = t["name"]
        crashes = 0
        while task_state(b, t)["remaining"] > 0:
            cmd = [PY, "-m", "adup.memguard", "--min-free-gb", str(b.get("min_free_gb", 10)),
                   "--log", os.path.join(b["log_dir"], f"{t['name']}.log"), "--",
                   "nice", "-n", "10", PY, "-m", "adup.make_pairs", "--sequences", t["specs"], "--out", t["out"],
                   *(["--limit", str(t["limit"])] if t.get("limit") else []), *t["args"]]
            log = os.path.join(b["log_dir"], f"{t['name']}.log")
            if os.path.exists(log):                            # memguard truncates its log: keep earlier runs
                with open(log) as a, open(log + ".all", "a") as z:
                    z.write(a.read())
            st["state"] = "running"
            save(f"{t['name']}：开始（已完成 {task_state(b, t)['done']} 条）")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while p.poll() is None:
                time.sleep(30)
                save()
            if p.returncode == 137:                            # stopped by memguard: wait for memory, then resume
                st["state"] = "waiting_memory"
                save(f"{t['name']}：可用内存不足，已暂停")
                while psutil.virtual_memory().available < b.get("resume_free_gb", 14) * 2 ** 30:
                    time.sleep(60)
                    save()
                save("内存恢复，继续")
            elif p.returncode != 0:
                crashes += 1
                save(f"{t['name']}：异常退出（代码 {p.returncode}），第 {crashes} 次")
                if crashes >= 3:
                    break
            else:
                break                                          # finished; ads that failed stay listed as failed
        s = task_state(b, t)
        save(f"{t['name']}：结束，完成 {s['done']} / {s['total']}，失败 {s['failed']}")
    st["state"], st["task"] = "finished", None
    save("全部完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("batch")
    ap.add_argument("--watch", type=int, default=0, help="status: refresh every N seconds")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.batch)
        return
    b = load(args.batch)
    while True:
        s = snapshot(b, args.batch)
        out = text(s)
        if args.watch:
            print("\033[2J\033[H" + out, flush=True)
            time.sleep(args.watch)
        else:
            print(out)
            break


if __name__ == "__main__":
    main()
