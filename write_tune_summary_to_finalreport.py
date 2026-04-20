#!/usr/bin/env python3
"""读取 run_tune_refined 输出目录中的 best_by_dataset.json，写入 ~/finalreport 下的 Markdown + JSON 副本。"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

_TZ_CN = timezone(timedelta(hours=8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.tune_dir.resolve()
    best_path = root / "best_by_dataset.json"
    if not best_path.is_file():
        raise SystemExit(f"Missing {best_path}")

    best = json.loads(best_path.read_text(encoding="utf-8"))
    final = Path.home() / "finalreport"
    final.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(_TZ_CN).strftime("%Y%m%d_%H%M%S")
    base = f"FMGAD_YelpChi_Cora_tune_best_{stamp}"
    gen_ts = datetime.now(_TZ_CN).strftime("%Y-%m-%d %H:%M:%S %z")

    md_path = final / f"{base}.md"
    json_copy = final / f"{base}_best_by_dataset.json"

    lines = [
        "# FMGAD：YelpChi / Cora 精细调参结果（多 seed 平均 AUC 最优）",
        "",
        f"- **报告生成时间**：{gen_ts}",
        f"- **调参输出目录**：`{root}`",
        "- **选优规则**：同一组超参（`cfg_id`）在 seeds `42, 0, 1, 2, 3` 上 **AUC 算术平均** 最高者（见 `run_tune_refined.py` 聚合逻辑）。",
        "",
        "## 各数据集最优",
        "",
        "| 数据集 | 平均 AUC | 参与 seed 数 | cfg_id |",
        "|--------|----------|--------------|--------|",
    ]

    for ds in ("yelpchi", "cora"):
        rec = best.get(ds)
        if not rec or "error" in rec:
            lines.append(f"| {ds} | - | - | 无有效运行 |")
            continue
        sm = rec.get("seed_runs") or []
        lines.append(
            "| {} | {:.6f} | {} | `{}` |".format(
                ds,
                float(rec.get("auc_mean", 0.0)),
                int(rec.get("num_seeds", len(sm))),
                str(rec.get("cfg_id", "")),
            )
        )

    lines.extend(["", "## 各 seed AUC 明细", ""])
    for ds in ("yelpchi", "cora"):
        rec = best.get(ds)
        lines.append(f"### {ds}")
        lines.append("")
        if not rec or "error" in rec:
            lines.append("（无）")
            lines.append("")
            continue
        for s in rec.get("seed_runs") or []:
            lines.append(
                "- seed {}: AUC={:.6f}, AP={:.6f}".format(
                    s.get("seed"),
                    float(s.get("auc", 0.0)),
                    float(s.get("ap", 0.0)),
                )
            )
        lines.append("")
        lines.append("**网格覆盖 `config`（不含基座 YAML 中与固定项重复键）**：")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(rec.get("config", {}), indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## 复现",
            "",
            f"- 完整运行记录：`{root / 'tuning_runs.json'}`",
            f"- 单次 run JSON：`{root / 'runs'}` 目录下 `*__seed*.json`（含 `seed` / `auc` / `full_config`）",
            f"- 最优合并 YAML：`{root / 'best_configs'}` 下 `*_best_refined.yaml`（若生成成功）",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(best_path, json_copy)
    print("Wrote:", md_path)
    print("Copied:", json_copy)


if __name__ == "__main__":
    main()
