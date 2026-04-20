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


def calibrate_polarity_tail_lcc(
    score: torch.Tensor,
    lcc: torch.Tensor,
    k_percent: float = 0.05,
    margin: float = 1.2,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    尾部 LCC 对比（无监督）：取分数最高 / 最低各约 k% 节点，比较两组的平均 LCC。
    若低分尾平均 LCC 显著高于高分尾（× margin），认为「低分端稠密塌陷」、极性反了，对 score 做 [0,1] 线性翻转。
    """
    with torch.no_grad():
        s = score.reshape(-1).float()
        lc = lcc.reshape(-1).float().to(s.device)
        n = int(s.numel())
        if n < 2 or lc.numel() != n:
            return score, False
        smin, smax = s.min(), s.max()
        if float(smax - smin) <= 1e-12:
            return score, False

        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)  # 避免上下尾重叠过多
        if k < 1:
            return score, False

        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        mean_lcc_top = lc[top_idx].mean()
        mean_lcc_bot = lc[bot_idx].mean()

        if verbose:
            print(
                f"[Tail-LCC] top-{k_percent*100:.1f}% mean LCC={float(mean_lcc_top):.4f}, "
                f"bot-{k_percent*100:.1f}% mean LCC={float(mean_lcc_bot):.4f} (margin={margin})",
                flush=True,
            )

        if float(mean_lcc_bot) > float(mean_lcc_top) * float(margin):
            if float(smax - smin) <= 1e-12:
                return -score, True
            return 1.0 - (score - smin) / (smax - smin), True
        return score, False


def calibrate_polarity_feature_anchor(
    score: torch.Tensor,
    raw_x: torch.Tensor,
    k_percent: float = 0.05,
    min_delta: float = 0.01,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    无监督特征锚点：用全体节点特征按维 median 作为「正常原型」，对分数最高/最低各约 k% 节点，
    计算其与原型的余弦相似度均值。若 sim_top - sim_bot > min_delta（高分尾更接近原型、低分尾更远），
    认为极性反了（正常被打出高分），对 score 做 [0,1] 线性翻转。仅使用 data.x，不用 GNN 嵌入。
    """
    with torch.no_grad():
        s = score.reshape(-1).float()
        x = raw_x.float()
        if x.dim() != 2 or int(x.size(0)) != int(s.numel()):
            return score, False
        smin, smax = s.min(), s.max()
        if float(smax - smin) <= 1e-12:
            return score, False
        n = int(s.numel())
        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        if k < 1:
            return score, False
        proto = torch.median(x, dim=0).values.unsqueeze(0)
        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        sim_top = F.cosine_similarity(x[top_idx], proto, dim=1, eps=1e-8).mean()
        sim_bot = F.cosine_similarity(x[bot_idx], proto, dim=1, eps=1e-8).mean()
        delta = float(sim_top - sim_bot)
        if verbose:
            print(
                f"[Feature-Anchor] sim_top={float(sim_top):.4f} sim_bot={float(sim_bot):.4f} delta={delta:.4f} (min_delta={min_delta})",
                flush=True,
            )
        if delta > float(min_delta):
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
