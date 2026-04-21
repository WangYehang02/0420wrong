#!/usr/bin/env python3
"""
在 Enron / Reddit / Disney / Books 上检验 SmoothGNN 秩一先验极性校准器的行为。

标签仅用于离线评估 AUC，不参与 calibrate_polarity_smoothgnn_anchor 决策。

检验项：
1) prior 单独作为分数的 AUC（先验与真实异常是否同向）。
2) 故意反极：score = -prior，看校准器是否翻转、AUC 是否回到与 prior 同向（机械一致性）。
3) 同向：score = prior，看是否不误翻转。
"""
from __future__ import annotations

import math
import os
import sys
import warnings

FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))
if FMGAD_ROOT not in sys.path:
    sys.path.insert(0, FMGAD_ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from pygod.metric.metric import eval_roc_auc
from pygod.utils import load_data

from utils import calibrate_polarity_smoothgnn_anchor, compute_smoothgnn_prior

DATASETS = ("enron", "reddit", "disney", "books")
K_PERCENT = 0.05
MARGIN = 1.1


def main() -> None:
    rows = []
    for name in DATASETS:
        try:
            data = load_data(name)
        except Exception as e:
            rows.append((name, "LOAD_FAIL", str(e)))
            print(f"[{name}] 加载失败: {e}", flush=True)
            continue

        x = data.x.float().cpu()
        ei = data.edge_index.long().cpu()
        y = data.y.view(-1).bool()

        with torch.no_grad():
            prior = compute_smoothgnn_prior(x, ei)

        auc_prior = float(eval_roc_auc(y, prior.cpu()))
        auc_neg = float(eval_roc_auc(y, (-prior).cpu()))

        s_wrong = -prior
        s_wrong_cal, flip_wrong = calibrate_polarity_smoothgnn_anchor(
            s_wrong,
            prior,
            k_percent=K_PERCENT,
            margin=MARGIN,
            verbose=True,
        )
        auc_wrong_cal = float(eval_roc_auc(y, s_wrong_cal.cpu()))

        s_right = prior.clone()
        s_right_cal, flip_right = calibrate_polarity_smoothgnn_anchor(
            s_right,
            prior,
            k_percent=K_PERCENT,
            margin=MARGIN,
            verbose=False,
        )
        auc_right_cal = float(eval_roc_auc(y, s_right_cal.cpu()))

        # 机械一致性：把 FMGAD 强行设成 -prior 时，校准后排序应与 prior 一致 → AUC 应回到 AUC(prior)
        mechanical = flip_wrong and math.isclose(auc_wrong_cal, auc_prior, rel_tol=0.0, abs_tol=1e-4)
        # 与标签同向：仅当 AUC(prior)>0.5 时，才说「先验与异常标签同向」
        prior_aligns_label = auc_prior >= 0.5

        noharm_ok = (not flip_right) and (abs(auc_right_cal - auc_prior) < 1e-3)

        print(
            f"\n[{name}] N={y.numel()} pos={int(y.sum())} | "
            f"AUC(prior)={auc_prior:.4f} AUC(-prior)={auc_neg:.4f}\n"
            f"  反极(-prior): flip={flip_wrong} → AUC(校准后)={auc_wrong_cal:.4f} "
            f"(相对误极前 Δ={auc_wrong_cal - auc_neg:+.4f})  "
            f"{'[机械恢复 OK: 与 prior 同向]' if mechanical else '[机械恢复异常]'}\n"
            f"  同向(prior): flip={flip_right} AUC(校准后)={auc_right_cal:.4f}  "
            f"{'[未误翻转]' if noharm_ok else '[同向时被误翻转或 AUC 漂移]'}  "
            f"先验与标签: {'同向(AUC≥0.5)' if prior_aligns_label else '反向(AUC<0.5)'}",
            flush=True,
        )
        rows.append(
            (
                name,
                auc_prior,
                auc_neg,
                flip_wrong,
                auc_wrong_cal,
                flip_right,
                auc_right_cal,
                mechanical,
                noharm_ok,
                prior_aligns_label,
            )
        )

    print("\n=== 汇总表 ===", flush=True)
    print(
        f"{'dataset':<10} {'AUC(p)':>8} {'AUC(-p)':>8} {'flip(-p)':>8} "
        f"{'AUC(cal)':>8} {'flip(p)':>7} {'机械OK':>6} {'同向无害':>8} {'先验∥标签':>10}",
        flush=True,
    )
    for r in rows:
        if len(r) == 3:
            print(f"{r[0]:<10} {r[1]:>10} {r[2]}", flush=True)
            continue
        (
            name,
            auc_prior,
            auc_neg,
            flip_wrong,
            auc_wrong_cal,
            flip_right,
            auc_right_cal,
            mechanical,
            noharm_ok,
            prior_aligns_label,
        ) = r
        print(
            f"{name:<10} {auc_prior:8.4f} {auc_neg:8.4f} {str(flip_wrong):>8} "
            f"{auc_wrong_cal:8.4f} {str(flip_right):>7} "
            f"{'Y' if mechanical else 'N':>6} "
            f"{'Y' if noharm_ok else 'N':>8} "
            f"{'Y' if prior_aligns_label else 'N':>10}",
            flush=True,
        )


if __name__ == "__main__":
    main()
