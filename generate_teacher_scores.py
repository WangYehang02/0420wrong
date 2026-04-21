#!/usr/bin/env python3
"""
离线生成完整 SmoothGNN(NAD) 全图异常分数，供 FMGAD ``smoothgnn_teacher_polarity`` 读取。

与内联 ``smoothgnn_full_reference.run_smoothgnn_nad_scores`` 使用同一训练逻辑；
输出默认写入本目录下 ``teacher_scores/{dataset}_smoothgnn_teacher.pt``。

用法示例：
  SMOOTHGNN_ROOT=/path/to/SmoothGNN \\
  python generate_teacher_scores.py --datasets weibo,disney,enron,books,reddit --seed 42 --nepoch 100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        type=str,
        default="weibo,disney,enron,books,reddit",
        help="逗号分隔，与 pygod/SmoothGNN load_data 名称一致",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nepoch", type=int, default=100)
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认: <本脚本目录>/teacher_scores",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "teacher_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    from smoothgnn_full_reference import run_smoothgnn_nad_scores

    for d in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        print(f"\n========== {d} ==========", flush=True)
        scores = run_smoothgnn_nad_scores(
            d,
            seed=args.seed,
            nepoch=args.nepoch,
            verbose=True,
        )
        out_path = out_dir / f"{d}_smoothgnn_teacher.pt"
        torch.save(scores, out_path)
        print(f"Saved {out_path} shape={tuple(scores.shape)}", flush=True)


if __name__ == "__main__":
    main()
