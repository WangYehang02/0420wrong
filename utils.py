"""
工具函数
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def compute_node_lcc_tensor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    无向图下每个节点的局部聚类系数 (NetworkX)，shape [N]，CPU float32。
    """
    import networkx as nx

    ei = edge_index.detach().cpu()
    data = Data(edge_index=ei, num_nodes=int(num_nodes))
    G = to_networkx(data, to_undirected=True)
    lcc_dict = nx.clustering(G)
    out = np.zeros(int(num_nodes), dtype=np.float32)
    for i in range(int(num_nodes)):
        out[i] = float(lcc_dict.get(i, 0.0))
    return torch.from_numpy(out)


def calibrate_polarity_lcc_spearman(
    score: torch.Tensor,
    lcc: torch.Tensor,
    threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    基于 Score 与局部聚类系数 LCC 的 Spearman 秩相关，无监督极性探针。
    rho < threshold 时对 score 做 [0,1] 线性翻转（与 flip_score 一致）。
    """
    with torch.no_grad():
        dev = score.device
        score_np = score.detach().cpu().numpy().ravel()
        lcc_np = lcc.detach().cpu().numpy().ravel()
        n = score_np.size
        if n < 2 or lcc_np.size != n:
            return score, False
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[LCC-Spearman] skip: zero variance in score", flush=True)
            return score, False
        rho, _p = spearmanr(score_np, lcc_np)
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            if verbose:
                print("[LCC-Spearman] skip: Spearman undefined (e.g. constant LCC)", flush=True)
            return score, False
        rho_f = float(rho)
        if verbose:
            print(f"[LCC-Spearman] rho(score, LCC)={rho_f:.4f} (threshold={threshold})", flush=True)
        if rho_f < threshold:
            smin, smax = score.min(), score.max()
            if float(smax - smin) <= 1e-12:
                return -score, True
            return 1.0 - (score - smin) / (smax - smin), True
        return score, False


def softmax_with_temperature(input, t=1, axis=-1):
    """
    带温度参数的 softmax 函数
    
    参数:
        input: 输入张量
        t: 温度参数，t > 1 时分布更平滑，t < 1 时分布更尖锐
        axis: 应用 softmax 的维度
    
    返回:
        归一化后的概率分布
    """
    return F.softmax(input / t, dim=axis)
