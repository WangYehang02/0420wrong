"""
调用 Baseline/SmoothGNN 完整 NAD 训练，得到节点级 anomaly 分数，供 FMGAD 极性校准。

注意：必须与 FMGAD 同进程的 ``utils`` 模块名解耦，否则会 import 到本仓库的 utils。
通过临时替换 ``sys.modules['utils']`` 等方式加载 SmoothGNN 包。

环境变量 SMOOTHGNN_ROOT 指向 SmoothGNN 根目录；默认
  <FMGAD 父级的父级>/Baseline/SmoothGNN
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import torch


def _smoothgnn_root() -> Path:
    env = os.environ.get("SMOOTHGNN_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    return (here.parent.parent / "Baseline" / "SmoothGNN").resolve()


def run_smoothgnn_nad_scores(
    dataset: str,
    seed: int,
    nepoch: int = 100,
    hidden_dim: int = 64,
    decay: float = 1e-6,
    verbose: bool = False,
) -> torch.Tensor:
    """
    训练 SmoothGNN (model.NAD)，按验证 AUC 最佳 epoch 保存全图 anomaly 分数 [N]（CPU float32）。
    """
    root = _smoothgnn_root()
    if not (root / "utils.py").is_file():
        raise FileNotFoundError(
            f"未找到 SmoothGNN 仓库（缺 {root}/utils.py）。请设置 SMOOTHGNN_ROOT 或把 Baseline/SmoothGNN 放在 {root}"
        )

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    # 与本仓库 ``import utils`` 冲突：临时卸下 FMGAD 的 utils
    _saved_utils = sys.modules.pop("utils", None)
    _saved_name = sys.modules.pop("name", None)
    sys.path.insert(0, str(root))
    try:
        os.chdir(str(root))
        import numpy as np
        from sklearn.metrics import roc_auc_score

        import name as sname  # noqa: E402
        import utils as sutils  # noqa: E402  # SmoothGNN/utils.py
        import model as sg_model  # noqa: E402
        from graphdata import Graph  # noqa: E402

        sutils.set_seed(seed)
        graph, features, labels, edge_index, index = sutils.load_data(dataset)
        n = len(features)
        m = int(edge_index.shape[1])
        index_np = np.asarray(index).reshape(-1).astype(np.int64)

        if dataset in sname.DATASETS:
            lr, hop, init, _paper_seed, eps = sname.set_paras(dataset)
        else:
            lr, hop, init, eps = 0.0001, 4, 0.05, 4e-3

        if verbose:
            print(
                f"[SmoothGNN-Full] dataset={dataset} n={n} m={m} hop={hop} lr={lr} eps={eps} nepoch={nepoch}",
                flush=True,
            )

        lap = sutils.get_lap(edge_index, n)
        infmatrix = sutils.get_infmatrix(edge_index, n, m, eps)
        graphdata = Graph(graph, features, labels, edge_index, infmatrix, lap, hop)

        nad = sg_model.NAD(features.shape[1], hidden_dim, 2, graphdata, init)
        optimizer = torch.optim.Adagrad(nad.parameters(), lr=lr, weight_decay=decay)

        best_auc = -1.0
        best_scores: Optional[torch.Tensor] = None

        labels_t = labels
        if labels_t.dim() > 1:
            labels_t = labels_t.squeeze(-1)

        for epoch in range(nepoch):
            nad.train()
            reconembed, anomalyembed = nad()
            loss = torch.mean(reconembed[index_np]) + torch.mean(anomalyembed[index_np])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nad.eval()
            with torch.no_grad():
                _, probs = nad()
                auc = roc_auc_score(
                    labels_t[index_np].cpu().numpy(),
                    probs[index_np].detach().cpu().numpy(),
                )
            if auc > best_auc:
                best_auc = float(auc)
                best_scores = probs.detach().float().cpu().clone()

            if verbose and (epoch % max(1, nepoch // 10) == 0 or epoch == nepoch - 1):
                print(
                    f"[SmoothGNN-Full] epoch {epoch}/{nepoch} loss={float(loss):.6f} auc={auc:.4f} best={best_auc:.4f}",
                    flush=True,
                )

        if best_scores is None:
            raise RuntimeError("SmoothGNN: no scores produced")
        if verbose:
            print(f"[SmoothGNN-Full] best val AUC (index)={best_auc:.4f}", flush=True)
        out = best_scores
    finally:
        # SmoothGNN utils.set_seed 会打开 deterministic + 关 cudnn，导致后续 FMGAD 在 GPU 上报错
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        os.chdir(old_cwd)
        sys.path[:] = old_path
        # 移除 SmoothGNN 子模块，恢复本仓库 utils
        for key in ("graphdata", "model", "utils", "name"):
            sys.modules.pop(key, None)
        if _saved_utils is not None:
            sys.modules["utils"] = _saved_utils
        if _saved_name is not None:
            sys.modules["name"] = _saved_name
        # 若其它代码仍持有对已删除模块的引用，后续 import utils 会重新加载 FMGAD utils
        importlib.invalidate_caches()

    return out
