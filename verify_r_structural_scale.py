#!/usr/bin/env python3
"""
诊断：use_multi_scale_residual=True 时 r_structural（度数差 expand）与 r_global/r_local 的量级对比。
不跑完整训练，仅用真实图结构与与 _build_z 一致的边处理（含可选虚拟邻居）+ 代表 AE 嵌入 h。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
import yaml

FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))
if FMGAD_ROOT not in sys.path:
    sys.path.insert(0, FMGAD_ROOT)

from encoder import compute_multi_scale_residuals, ResidualChannelAttention
from res_flow_gad import ResFlowGAD, _add_virtual_knn_edges


def _hid_dim_from_features(in_dim: int) -> int:
    return 2 ** int(math.log2(in_dim) - 1)


def stats(name: str, t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "name": name,
        "mean_abs": float(t.abs().mean()),
        "std": float(t.std()),
        "max_abs": float(t.abs().max()),
        "p99_abs": float(torch.quantile(t.abs().flatten(), 0.99)),
    }


def row_l2(t: torch.Tensor) -> torch.Tensor:
    return torch.norm(t, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)

    datasets = ["weibo", "disney", "reddit", "books", "enron"]
    print("device:", dev, flush=True)

    for dset in datasets:
        cfg_path = os.path.join(FMGAD_ROOT, "configs", f"{dset}.yaml")
        if not os.path.isfile(cfg_path):
            print(f"\n=== {dset} SKIP (no {cfg_path}) ===", flush=True)
            continue
        cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)
        use_vn = bool(cfg.get("use_virtual_neighbors", True))

        loader = ResFlowGAD(verbose=False)
        try:
            data = loader._load_dataset(dset)
        except Exception as e:
            print(f"\n=== {dset} LOAD FAILED: {e} ===", flush=True)
            continue

        n = data.x.size(0)
        in_dim = data.num_node_features
        hid_dim = cfg.get("hid_dim")
        if hid_dim is None:
            hid_dim = _hid_dim_from_features(in_dim)

        edge_index = data.edge_index.to(dev)
        # 代表 AE 嵌入：LayerNorm 后数值与常见归一化表征同量级
        h = torch.randn(n, hid_dim, device=dev)
        h = F.layer_norm(h, (hid_dim,))

        ei = edge_index
        if use_vn:
            ei = _add_virtual_knn_edges(
                ei, h, int(cfg.get("virtual_degree_threshold", 5)), int(cfg.get("virtual_k", 5)), dev
            )

        r_g, r_l, r_s, deg = compute_multi_scale_residuals(h, ei)

        # 纯图度数差（与 h 无关），用于验证「50～100」说法
        src, dst = ei[0], ei[1]
        deg_val = torch.zeros(n, device=dev, dtype=torch.float32)
        deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
        deg_src = deg_val[src]
        neigh_deg_sum = torch.zeros(n, device=dev, dtype=torch.float32)
        neigh_deg_sum.index_add_(0, dst, deg_src)
        neigh_deg_mean = (neigh_deg_sum / deg_clamped.squeeze(1)).unsqueeze(1)
        raw_diff = (deg_val.unsqueeze(1) - neigh_deg_mean).abs()

        sg, sl, ss = stats("r_global", r_g), stats("r_local", r_l), stats("r_structural", r_s)
        rg_l2 = row_l2(r_g)
        rl_l2 = row_l2(r_l)
        rs_l2 = row_l2(r_s)

        attn = ResidualChannelAttention(3, hid_dim).to(dev)
        with torch.no_grad():
            fused = attn([r_g, r_l, r_s])
            cat = torch.cat([r_g, r_l, r_s], dim=1)
            # 与 ResidualChannelAttention 内部一致，看未训练时各通道对 logits 的「输入能量」占比
            w0 = attn.w[0].weight
            # 粗略：各块对 linear 输入的贡献范数（仅作参考）
            part = hid_dim
            chunks = [cat[:, i * part : (i + 1) * part] for i in range(3)]
            chunk_energy = [float(c.pow(2).sum().sqrt()) for c in chunks]

        ratio_mean = ss["mean_abs"] / (sg["mean_abs"] + 1e-8)
        ratio_l2_med = float(rs_l2.median() / (rg_l2.median() + 1e-8))

        print(f"\n=== {dset} (N={n}, hid_dim={hid_dim}, virtual_neighbors={use_vn}, |E|={ei.size(1)}) ===", flush=True)
        print(
            f"  raw_degree_diff |deg-mean(deg_nbr)|:  max={float(raw_diff.max()):.2f}  "
            f"p99={float(torch.quantile(raw_diff, 0.99)):.2f}  mean={float(raw_diff.mean()):.2f}",
            flush=True,
        )
        print(f"  {sg}", flush=True)
        print(f"  {sl}", flush=True)
        print(f"  {ss}", flush=True)
        print(
            f"  ratio mean_abs(r_s)/mean_abs(r_g) = {ratio_mean:.2f}  ;  "
            f"median L2(r_s)/median L2(r_g) = {ratio_l2_med:.2f}",
            flush=True,
        )
        print(f"  concat chunk L2 energy [global, local, structural] = {chunk_energy}", flush=True)
        print(
            f"  fused vs r_g: mean_abs(fused)/mean_abs(r_g)={float(fused.abs().mean()/(r_g.abs().mean()+1e-8)):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
