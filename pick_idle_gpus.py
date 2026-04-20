#!/usr/bin/env python3
"""按利用率/显存占用挑选较空闲 GPU，打印为空格分隔的索引列表（供 bash 展开）。"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--want", type=int, default=8, help="最多返回几个 GPU")
    p.add_argument("--util-max", type=float, default=25.0, help="利用率上限 (%)")
    p.add_argument("--mem-max-mib", type=float, default=15000.0, help="已用显存上限 (MiB)")
    p.add_argument("--fallback", type=str, default="0,1,2,3", help="无满足条件时的回退列表，逗号分隔")
    args = p.parse_args()

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as e:
        print(args.fallback.replace(",", " "), file=sys.stdout)
        print(f"[pick_idle_gpus] nvidia-smi failed ({e}), using fallback", file=sys.stderr)
        return

    rows: list[tuple[float, float, int]] = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        idx, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
        rows.append((util, mem, idx))
    rows.sort(key=lambda x: (x[0], x[1]))

    picked: list[int] = []
    for util, mem, idx in rows:
        if util <= args.util_max and mem <= args.mem_max_mib:
            picked.append(idx)
        if len(picked) >= args.want:
            break

    if len(picked) < 2:
        picked = [int(x) for x in args.fallback.split(",") if x.strip()][: args.want]

    print(" ".join(str(i) for i in picked[: args.want]))


if __name__ == "__main__":
    main()
