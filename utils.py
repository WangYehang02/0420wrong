"""
工具函数
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx, to_undirected


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


def calibrate_polarity_spearman_reference(
    score: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    与 ``calibrate_polarity_lcc_spearman`` 同构，将第二路标量换为任意「参考异常分数」（如完整 SmoothGNN NAD 输出）。
    rho(score, reference) < threshold 时对 score 做 [0,1] 线性翻转。
    """
    with torch.no_grad():
        score_np = score.detach().cpu().numpy().ravel()
        ref_np = reference.detach().cpu().numpy().ravel()
        n = score_np.size
        if n < 2 or ref_np.size != n:
            return score, False
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[Ref-Spearman] skip: zero variance in score", flush=True)
            return score, False
        rho, _p = spearmanr(score_np, ref_np)
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            if verbose:
                print("[Ref-Spearman] skip: Spearman undefined", flush=True)
            return score, False
        rho_f = float(rho)
        if verbose:
            print(f"[Ref-Spearman] rho(score, ref)={rho_f:.4f} (threshold={threshold})", flush=True)
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


def compute_local_global_l2_distances(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    undirected: bool = True,
    include_self: bool = True,
) -> torch.Tensor:
    """
    局部上下文 = 邻居特征均值（可选无向化、可选含自身，避免孤立点退化为零向量）。
    全局上下文 = 全图均值向量。
    返回每个节点的 ||z_local - z_global||_2。
    """
    ei = to_undirected(edge_index.long()) if undirected else edge_index.long()
    zf = z.float()
    n = zf.size(0)
    src, dst = ei[0], ei[1]
    neigh_sum = torch.zeros_like(zf)
    neigh_sum.index_add_(0, dst, zf[src])
    deg = torch.zeros(n, device=zf.device, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    if include_self:
        neigh_sum = neigh_sum + zf
        deg = deg + 1.0
    deg_u = deg.clamp_min(1.0).unsqueeze(1)
    z_local = neigh_sum / deg_u
    z_global = zf.mean(dim=0, keepdim=True).expand(n, -1)
    return torch.norm(z_local - z_global, p=2, dim=1)


def calibrate_polarity_smooth_discrepancy(
    score: torch.Tensor,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    mode: str = "spearman",
    spearman_threshold: float = -0.05,
    k_percent: float = 0.05,
    tail_margin: float = 1.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    基于「局部 − 全局」几何差异的极性探针（灵感来自 SmoothGNN，但并非其训练后的网络输出）。

    重要说明（为何 Enron/Reddit 上旧版 tail 会害死人）：
    - SmoothGNN 论文里局部/全局是在**训练后的表示**上比较的；仅用原始 x 时，该距离与异常标签
      在不少数据集上几乎无关（甚至 AUC<0.5）。此时仍用「尾均值谁大」去裁决 FMGAD 极性，会频繁误翻转。
    - PyG 中 Enron 等图为**有向**边，旧实现只按 dst 聚合会破坏 1-hop 语义；孤立点旧实现把局部特征置 0，
      会人为制造巨大距离，使低分尾均值虚高。

    默认 ``mode="spearman"``：计算 rho(score, distance)，若 rho < spearman_threshold 则翻转
    （与 ``calibrate_polarity_lcc_spearman`` 同构）；rho 接近 0 时不翻转，避免在弱相关数据上乱动分数。

    ``mode="tail"``：保留旧版尾均值对比；可选 tail_margin>1 使翻转更保守（类似 tail-LCC）。
    """
    with torch.no_grad():
        s = score.reshape(-1).float()
        n = int(s.numel())
        if n < 2:
            return score, False

        distances = compute_local_global_l2_distances(z, edge_index)
        mode_l = str(mode).strip().lower()

        if mode_l == "spearman":
            score_np = s.detach().cpu().numpy().ravel()
            dist_np = distances.detach().cpu().numpy().ravel()
            if np.std(score_np) <= 1e-12 or np.std(dist_np) <= 1e-12:
                if verbose:
                    print("[Smooth-Probe] skip: near-constant score or distance", flush=True)
                return score, False
            rho, _p = spearmanr(score_np, dist_np)
            if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                if verbose:
                    print("[Smooth-Probe] skip: Spearman undefined", flush=True)
                return score, False
            rho_f = float(rho)
            if verbose:
                print(
                    f"[Smooth-Probe] rho(score,dist)={rho_f:.4f} (threshold={spearman_threshold})",
                    flush=True,
                )
            if rho_f < float(spearman_threshold):
                smin, smax = s.min(), s.max()
                if float(smax - smin) <= 1e-12:
                    return -score, True
                return 1.0 - (score - smin) / (smax - smin), True
            return score, False

        # --- legacy: tail mean comparison ---
        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        mean_dist_top = distances[top_idx].mean()
        mean_dist_bot = distances[bot_idx].mean()
        if verbose:
            print(
                f"[Smooth-Probe/tail] dist_top={float(mean_dist_top):.4f}, dist_bot={float(mean_dist_bot):.4f} "
                f"(margin={tail_margin})",
                flush=True,
            )
        if float(mean_dist_bot) > float(mean_dist_top) * float(tail_margin):
            smin, smax = s.min(), s.max()
            if float(smax - smin) <= 1e-12:
                return -score, True
            return 1.0 - (score - smin) / (smax - smin), True
        return score, False


def compute_smoothgnn_local_prior(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    局部拉普拉斯型平滑残差：||x_i - mean(neighbors(i))||_2（入边聚合）。
    不构造 N×N 矩阵；局部差异越大越违背同质性平滑假设。
    """
    with torch.no_grad():
        xf = x.float()
        src, dst = edge_index[0], edge_index[1]
        n = xf.size(0)
        neigh_sum = torch.zeros_like(xf)
        neigh_sum.index_add_(0, dst, xf[src])
        deg = torch.zeros(n, device=xf.device, dtype=xf.dtype)
        deg.index_add_(0, dst, torch.ones_like(xf[src, 0]))
        deg_u = deg.clamp_min(1.0).unsqueeze(-1)
        neigh_mean = neigh_sum / deg_u
        return torch.norm(xf - neigh_mean, p=2, dim=1)


def calibrate_polarity_robust(
    score: torch.Tensor,
    local_prior: torch.Tensor,
    k_percent: float = 0.05,
    margin: float = 1.05,
    spearman_threshold: float = -0.1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    稳健极性校准：Spearman 秩相关（对极端分值不敏感）+ 尾部分组的中位数对比（抗离群）。
    先验与分数显著负相关时认为极性反了；否则再用尾部分组中位数判定。
    """
    with torch.no_grad():
        s = score.reshape(-1).float()
        prior = local_prior.reshape(-1).float().to(s.device)
        n = int(s.numel())
        if n < 2 or prior.numel() != n:
            return score, False
        smin, smax = s.min(), s.max()
        span = float(smax - smin)
        if span <= 1e-12:
            return score, False

        s_np = s.detach().cpu().numpy().ravel()
        prior_np = prior.detach().cpu().numpy().ravel()
        if np.std(s_np) <= 1e-12 or np.std(prior_np) <= 1e-12:
            if verbose:
                print("[Robust-Probe] skip Spearman: near-constant score or prior", flush=True)
            rho_f = None
        else:
            rho, _ = spearmanr(s_np, prior_np)
            if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                rho_f = None
            else:
                rho_f = float(rho)

        if rho_f is not None and rho_f < float(spearman_threshold):
            if verbose:
                print(f"[Robust-Probe] rho(score,prior)={rho_f:.3f} < {spearman_threshold}. Flipping.", flush=True)
            return 1.0 - (s - smin) / (smax - smin + 1e-8), True

        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        if k < 1:
            return score, False

        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        prior_top_med = torch.median(prior[top_idx])
        prior_bot_med = torch.median(prior[bot_idx])

        if verbose:
            print(
                f"[Robust-Probe] tail medians: top-k prior={float(prior_top_med):.4f}, "
                f"bot-k prior={float(prior_bot_med):.4f} (margin={margin})",
                flush=True,
            )

        if float(prior_bot_med) > float(prior_top_med) * float(margin):
            if verbose:
                print("[Robust-Probe] bot tail median prior > top × margin. Flipping.", flush=True)
            return 1.0 - (s - smin) / (smax - smin + 1e-8), True

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


def compute_smoothgnn_prior(
    x: torch.Tensor, edge_index: torch.Tensor, eps: float = 4e-3
) -> torch.Tensor:
    """
    极轻量化：与 SmoothGNN ``get_infmatrix`` 的秩一无穷平滑矩阵 P^\\infty = d d^\\top 等价，
    用结合律计算 x_\\infty = P^\\infty X = d (d^\\top X)，避免构造 N×N 稠密阵。

    返回每个节点 ||X_i - x_{\\infty,i}||_2（越大越违背图平滑驻留假设，越可能为异常）。
    """
    with torch.no_grad():
        n = int(x.size(0))
        m = int(edge_index.size(1))
        xf = x.float()
        deg = degree(edge_index[0], n, dtype=torch.float32) + 1.0
        deg_sqrt = torch.sqrt(deg / (2.0 * float(m) + float(n)))
        deg_sqrt = torch.where(deg_sqrt < eps, torch.zeros_like(deg_sqrt), deg_sqrt)
        inner = torch.matmul(deg_sqrt.unsqueeze(0), xf)
        x_inf = deg_sqrt.unsqueeze(1) * inner
        return torch.norm(xf - x_inf, p=2, dim=1)


def calibrate_polarity_smoothgnn_anchor(
    score: torch.Tensor,
    smooth_prior: torch.Tensor,
    k_percent: float = 0.05,
    margin: float = 1.1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """
    用 SmoothGNN 平滑先验做无监督极性校准：比较 FMGAD 分数最高/最低各约 k% 节点在先验上的均值。
    若低分尾先验显著更高，认为 FMGAD 极性反了，对 score 做与历史一致的 [0,1] 线性翻转。
    """
    with torch.no_grad():
        s = score.reshape(-1).float()
        prior = smooth_prior.reshape(-1).float().to(s.device)
        n = int(s.numel())
        if n < 2 or prior.numel() != n:
            return score, False
        smin, smax = s.min(), s.max()
        if float(smax - smin) <= 1e-12:
            return score, False

        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        if k < 1:
            return score, False

        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        prior_top = prior[top_idx].mean()
        prior_bot = prior[bot_idx].mean()

        if verbose:
            print(
                f"[SmoothGNN-Probe] Top-{float(k_percent)*100:.1f}% prior={float(prior_top):.4f}, "
                f"Bot prior={float(prior_bot):.4f} (margin={margin})",
                flush=True,
            )

        if float(prior_bot) > float(prior_top) * float(margin):
            if float(smax - smin) <= 1e-12:
                return -score, True
            return 1.0 - (score - smin) / (smax - smin), True

        return score, False
