#!/usr/bin/env python3
"""
在多个数据集 × 多个 seed 上并行跑 main_train.py，自动挑选空闲 GPU。

示例：
  python run_five_dsets_multiseed_parallel.py --conda-env fmgad
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FMGAD_ROOT = Path(__file__).resolve().parent


@dataclass
class Job:
    dataset: str
    seed: int
    gpu: int


def _pick_idle_gpus(want: int, util_max: float, mem_max_mib: float, fallback: str) -> list[int]:
    cmd = [
        sys.executable,
        str(FMGAD_ROOT / "pick_idle_gpus.py"),
        "--want",
        str(want),
        "--util-max",
        str(util_max),
        "--mem-max-mib",
        str(mem_max_mib),
        "--fallback",
        fallback,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return [int(x) for x in out.split() if x.strip()]


def _conda_python(env: str) -> list[str]:
    # 直接调用 env 内 python，避免 conda run 额外进程与 PATH 问题
    for root in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path("/opt/conda")):
        p = root / "envs" / env / "bin" / "python"
        if p.is_file():
            return [str(p)]
    base = os.environ.get("CONDA_EXE", "")
    if base:
        return ["conda", "run", "-n", env, "--no-capture-output", "python"]
    raise FileNotFoundError(f"找不到 conda 环境 {env} 的 python（试过 miniconda3/anaconda3/envs/{env}）")


def _config_path(dataset: str) -> Path:
    p = FMGAD_ROOT / "configs" / f"{dataset}_best.yaml"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        type=str,
        default="weibo,disney,enron,books,reddit",
        help="逗号分隔；需存在 configs/{name}_best.yaml",
    )
    p.add_argument("--seeds", type=str, default="42,0,1,2,3", help="逗号分隔")
    p.add_argument("--conda-env", type=str, default="fmgad")
    p.add_argument("--num-trial", type=int, default=None, help="默认读 yaml")
    p.add_argument("--timeout-sec", type=int, default=0, help="0 表示不限制单任务超时")
    p.add_argument("--output-dir", type=str, default="", help="默认 runs/parallel_<UTC时间戳>")
    p.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="同时跑几个任务；0 表示与可用 GPU 数相同（每卡一进程）",
    )
    p.add_argument("--gpus", type=str, default="", help="手动指定 GPU，如 '0,1,2'；空则自动探测")
    p.add_argument("--util-max", type=float, default=30.0)
    p.add_argument("--mem-max-mib", type=float, default=20000.0)
    args = p.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    if args.gpus.strip():
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = _pick_idle_gpus(8, args.util_max, args.mem_max_mib, "0,1,2,3")
    if not gpus:
        print("没有可用 GPU", file=sys.stderr)
        return 1

    max_par = args.max_parallel if args.max_parallel > 0 else len(gpus)
    max_par = min(max_par, len(gpus))
    if max_par < 1:
        max_par = 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.output_dir) if args.output_dir else FMGAD_ROOT / "runs" / f"parallel_{ts}"
    runs_dir = out_root / "runs"
    logs_dir = out_root / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    py = _conda_python(args.conda_env)

    jobs: list[Job] = []
    for ds in datasets:
        _config_path(ds)
        for sd in seeds:
            jobs.append(Job(dataset=ds, seed=sd, gpu=-1))

    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fmfad_root": str(FMGAD_ROOT),
        "conda_env": args.conda_env,
        "python_cmd": py,
        "datasets": datasets,
        "seeds": seeds,
        "gpus": gpus,
        "max_parallel": max_par,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # 每个并发任务独占一张卡；slot_gpus 为实际会用到的 GPU 子集
    slot_gpus = gpus[:max_par]
    free_gpus: list[int] = list(slot_gpus)

    running: list[tuple[subprocess.Popen, Job, Any]] = []

    def start_one(j: Job) -> None:
        if not free_gpus:
            raise RuntimeError("内部错误：无空闲 GPU 却尝试启动任务")
        gpu = free_gpus.pop(0)
        j = Job(dataset=j.dataset, seed=j.seed, gpu=gpu)
        result_file = runs_dir / f"{j.dataset}__seed{j.seed}.json"
        log_file = logs_dir / f"{j.dataset}__seed{j.seed}.gpu{j.gpu}.log"
        # 父进程设置 CUDA_VISIBLE_DEVICES，子进程内应使用逻辑 cuda:0（与 main_train 早初始化一致）
        cmd = py + [
            str(FMGAD_ROOT / "main_train.py"),
            "--config",
            str(_config_path(j.dataset)),
            "--device",
            "0",
            "--seed",
            str(j.seed),
            "--result-file",
            str(result_file),
        ]
        if args.num_trial is not None:
            cmd += ["--num_trial", str(args.num_trial)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(j.gpu)
        env["FMGAD_MODEL_ROOT"] = str(out_root / "model_cache")
        if "SMOOTHGNN_ROOT" not in env:
            _sg = Path.home() / "Baseline" / "SmoothGNN"
            if _sg.is_dir():
                env["SMOOTHGNN_ROOT"] = str(_sg)
        # 与固定 exp_tag 组合时按 seed 隔离目录，避免多进程写坏 dm_self.pt / ae ckpt
        env["FMGAD_RUN_TAG_SUFFIX"] = f"seed{j.seed}"
        env["TMPDIR"] = str(out_root / "tmp" / f"{j.dataset}_seed{j.seed}")
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        env["XDG_CACHE_HOME"] = str(out_root / "xdg" / f"{j.dataset}_seed{j.seed}")
        Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

        lf = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(FMGAD_ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((proc, j, lf))
        print(f"START {j.dataset} seed={j.seed} gpu={j.gpu} log={log_file}", flush=True)

    job_iter = iter(jobs)
    try:
        for _ in range(min(max_par, len(jobs))):
            start_one(next(job_iter))
    except StopIteration:
        pass

    failures: list[dict[str, Any]] = []
    while running:
        time.sleep(0.4)
        for item in list(running):
            proc, j, lf = item
            code = proc.poll()
            if code is None:
                continue
            lf.close()
            running.remove(item)
            free_gpus.append(j.gpu)
            ok = code == 0 and (runs_dir / f"{j.dataset}__seed{j.seed}.json").is_file()
            status = "OK" if ok else f"FAIL(code={code})"
            print(f"END {j.dataset} seed={j.seed} gpu={j.gpu} {status}", flush=True)
            if not ok:
                failures.append({"dataset": j.dataset, "seed": j.seed, "gpu": j.gpu, "returncode": code})
            try:
                nxt = next(job_iter)
                start_one(nxt)
            except StopIteration:
                pass

    # 汇总
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        for sd in seeds:
            rf = runs_dir / f"{ds}__seed{sd}.json"
            if not rf.is_file():
                rows.append({"dataset": ds, "seed": sd, "auc": None, "ap": None, "error": "missing_json"})
                continue
            try:
                data = json.loads(rf.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                rows.append({"dataset": ds, "seed": sd, "auc": None, "ap": None, "error": str(e)})
                continue
            rows.append(
                {
                    "dataset": ds,
                    "seed": sd,
                    "auc": data.get("auc", data.get("auc_mean")),
                    "ap": data.get("ap_mean", data.get("ap")),
                    "time_sec": data.get("time_sec"),
                }
            )

    by_ds: dict[str, list[float]] = {}
    for r in rows:
        if r.get("auc") is None:
            continue
        by_ds.setdefault(str(r["dataset"]), []).append(float(r["auc"]))

    summary: dict[str, Any] = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_root),
        "failures": failures,
        "per_run": rows,
        "mean_auc_by_dataset": {
            ds: sum(v) / len(v) if v else None for ds, v in sorted(by_ds.items(), key=lambda x: x[0])
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 多数据集 × 多 seed 并行结果\n", f"目录: `{out_root}`\n\n", "| dataset | mean AUC | n |\n|---|---|---|\n"]
    for ds in sorted(by_ds.keys()):
        v = by_ds[ds]
        lines.append(f"| {ds} | {sum(v)/len(v):.4f} | {len(v)} |\n")
    lines.append("\n## 逐次\n\n| dataset | seed | AUC | AP |\n|---|---|---|---|\n")
    for r in rows:
        auc_s = f"{float(r['auc']):.4f}" if r.get("auc") is not None else ""
        ap_s = f"{float(r['ap']):.4f}" if r.get("ap") is not None else ""
        lines.append(f"| {r.get('dataset','')} | {r.get('seed','')} | {auc_s} | {ap_s} |\n")
    (out_root / "summary.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== summary ===", flush=True)
    print(json.dumps(summary["mean_auc_by_dataset"], indent=2, ensure_ascii=False))
    print(f"Wrote {out_root / 'summary.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
