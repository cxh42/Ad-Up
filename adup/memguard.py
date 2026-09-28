"""Run a command under a memory guard: report the peak RSS of its whole process tree and stop it before the machine
runs out of memory.

The local PC has 32 GB and freezes when a job fills it (pair generation starts many ffmpeg encoders and decoders), so
local jobs run under this guard: it samples the process tree twice a second and terminates it when the system's
available memory drops below --min-free-gb. Jobs that resume (make_pairs skips finished ads) can simply be rerun.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.memguard [--min-free-gb 10] [--log out.log] -- <command ...>
Exit code: the command's, or 137 when the guard stopped it.
"""

import argparse
import contextlib
import subprocess
import sys
import time

import psutil


def tree_rss(p):
    total = 0
    for q in [p, *p.children(recursive=True)]:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total


def stop(p):
    procs = [p, *p.children(recursive=True)]
    for q in procs:
        try:
            q.terminate()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="stop the command below this much available memory")
    ap.add_argument("--log", help="write the command's stdout and stderr here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    with open(args.log, "w") if args.log else contextlib.nullcontext() as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None)
        p, peak, stopped, t0 = psutil.Process(proc.pid), 0, False, time.time()
        while proc.poll() is None:
            peak = max(peak, tree_rss(p))
            avail = psutil.virtual_memory().available
            if avail < args.min_free_gb * 2 ** 30:
                print(f"memguard: available memory {avail / 2 ** 30:.1f} GB < {args.min_free_gb} GB, stopping", flush=True)
                stop(p)
                stopped = True
                break
            time.sleep(0.5)
        code = proc.wait()
    print(f"memguard: peak RSS {peak / 2 ** 30:.2f} GB, {time.time() - t0:.0f} s, exit {137 if stopped else code}", flush=True)
    sys.exit(137 if stopped else code)


if __name__ == "__main__":
    main()
