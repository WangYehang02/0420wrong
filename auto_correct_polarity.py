#!/usr/bin/env python3
"""
无监督极性纠正：用 Isolation Forest 在节点特征上的分数作为锚点，
与 FMGAD 分数算 Spearman；rho < 0 时对 FMGAD 分数做线性 [0,1] 翻转。
决策仅使用 data.x 与模型分数，不使用标签。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))
if FMGAD_ROOT not in sys.path:
    sys.path.insert(0, FMGAD_ROOT)


def compute_iforest_anchor_scores(
    x: Union[torch.Tensor, np.ndarray],
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """在特征上拟合 iForest；返回「越大越异常」的锚点分数（numpy 1D）。"""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().float().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[0] < 2:
        raise ValueError("x 应为 [N, F] 且 N>=2")
    clf = IsolationForest(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
    )
    clf.fit(x_np)
    # sklearn: score_samples 越大越像正常点 → 取负与「异常高分」约定一致
    return -clf.score_samples(x_np).astype(np.float64)


def correct_scores_iforest_anchor(
    fmgad_scores: torch.Tensor,
    x: torch.Tensor,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float, bool]:
    """
    根据 Spearman(FMGAD, iForest_anchor) 是否 < 0 决定是否翻转 fmgad_scores。
    返回 (纠正后的分数 [同 device/dtype 与输入], rho, 是否翻转)。
    """
    device = fmgad_scores.device
    dtype = fmgad_scores.dtype
    s = fmgad_scores.detach().reshape(-1).float().cpu().numpy()
    anchor = compute_iforest_anchor_scores(x, n_estimators=n_estimators, random_state=random_state)

    rho, _p = spearmanr(s, anchor)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        rho_f = 0.0
        flipped = False
        out = s
        if verbose:
            print("[iforest_polarity] Spearman 无效（常数列等），不翻转。", flush=True)
    else:
        rho_f = float(rho)
        flipped = rho_f < 0.0
        if flipped:
            smin, smax = float(s.min()), float(s.max())
            if smax - smin > 1e-8:
                out = 1.0 - (s - smin) / (smax - smin)
            else:
                out = -s
            if verbose:
                print(
                    f"[iforest_polarity] Spearman rho={rho_f:.4f} < 0，已翻转 FMGAD 分数。",
                    flush=True,
                )
        else:
            out = s.copy()
            if verbose:
                print(
                    f"[iforest_polarity] Spearman rho={rho_f:.4f} >= 0，保持原方向。",
                    flush=True,
                )

    out_t = torch.from_numpy(out.astype(np.float32)).to(device=device, dtype=dtype)
    return out_t, rho_f, flipped


def correct_scores_with_anchor(
    fmgad_scores: torch.Tensor,
    data_x: torch.Tensor,
    dataset_name: str,
    labels: Optional[torch.Tensor] = None,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[torch.Tensor, bool, float]:
    """
    与用户文档一致的命名封装；labels 仅用于可选诊断打印，不参与翻转决策。
    """
    corrected, rho, flipped = correct_scores_iforest_anchor(
        fmgad_scores,
        data_x,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=verbose,
    )
    if verbose:
        print(f"[{dataset_name}] FMGAD vs iForest-anchor Spearman rho = {rho:.4f}", flush=True)

    if labels is not None:
        from sklearn.metrics import roc_auc_score

        y = labels.detach().cpu().numpy().reshape(-1)
        s0 = fmgad_scores.detach().cpu().numpy().reshape(-1)
        s1 = corrected.detach().cpu().numpy().reshape(-1)
        anc = compute_iforest_anchor_scores(data_x, n_estimators=n_estimators, random_state=random_state)
        print(f"  [诊断] 原始 FMGAD AUC: {roc_auc_score(y, s0):.4f}", flush=True)
        print(f"  [诊断] iForest 锚点 AUC: {roc_auc_score(y, anc):.4f}", flush=True)
        print(f"  [诊断] 纠正后 FMGAD AUC: {roc_auc_score(y, s1):.4f}", flush=True)

    return corrected, flipped, rho


def _demo_one_dataset(dset: str, seed: int) -> None:
    from res_flow_gad import ResFlowGAD

    dummy = ResFlowGAD(verbose=False)
    data = dummy._load_dataset(dset)
    x = data.x
    n = x.size(0)
    rng = np.random.default_rng(seed)
    sim = rng.standard_normal(n).astype(np.float32)
    if hasattr(data, "y") and data.y is not None:
        y = data.y.view(-1).bool().cpu().numpy()
        sim[y] -= 2.0
    scores = torch.from_numpy(sim)
    correct_scores_with_anchor(
        scores,
        x,
        dset,
        labels=data.y if hasattr(data, "y") else None,
        random_state=42,
        verbose=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="iForest 锚点极性纠正（演示 / 诊断）")
    p.add_argument("--demo", action="store_true", help="在若干数据集上用模拟反极性分数跑一遍诊断")
    p.add_argument(
        "--datasets",
        type=str,
        default="disney,enron,books,reddit",
        help="逗号分隔（仅 --demo）",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.demo:
        for d in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            print("=" * 60)
            print(f"演示数据集: {d}")
            try:
                _demo_one_dataset(d, args.seed)
            except Exception as e:
                print(f"跳过 {d}: {e}", flush=True)
        return 0

    print("用法: python auto_correct_polarity.py --demo", flush=True)
    print("训练评估请用 configs 中 iforest_anchor_polarity: true，由 res_flow_gad.sample() 自动调用。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
