import os
import sys


def _cuda_visible_devices_from_argv_early() -> None:
    """在 import torch 之前锁定可见 GPU，避免多进程并行时全部先初始化到物理 GPU 0。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    argv = sys.argv
    for i in range(len(argv) - 1):
        if argv[i] == "--device":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1])
            return


_cuda_visible_devices_from_argv_early()

import time
import yaml
import argparse
import json

from res_flow_gad import ResFlowGAD


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "weibo_best.yaml"))
    parser.add_argument("--result-file", type=str, default=None, help="Optional: write metrics JSON to this file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_trial", type=int, default=None, help="Number of trials (default from config or 3)")
    return parser.parse_args()


def _set_seed(seed: int):
    """固定随机种子，使结果可复现"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = get_arguments()
    _set_seed(args.seed)
    print("Random seed:", args.seed, flush=True)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    dset = cfg["dataset"]

    # 和 v3 一样：如果配置里给了 ae_alpha=0.0，自动回退到一个安全值，避免 NaN
    ae_alpha_cfg = cfg.get("ae_alpha", 0.8)
    if ae_alpha_cfg == 0.0:
        ae_alpha_cfg = 0.9

    model = ResFlowGAD(
        hid_dim=cfg.get("hid_dim") if cfg.get("hid_dim") else None,
        ae_dropout=cfg["ae_dropout"],
        ae_lr=cfg["ae_lr"],
        ae_alpha=ae_alpha_cfg,
        proto_alpha=cfg.get("proto_alpha", 0.01),
        weight=cfg.get("weight", 1.0),
        residual_scale=float(cfg.get("residual_scale", 10.0)),
        sample_steps=int(cfg.get("sample_steps", 100)),
        verbose=True,
        use_nll_score=cfg.get("use_nll_score", False),
        use_energy_score=cfg.get("use_energy_score", False),
        use_guided_recon=cfg.get("use_guided_recon", False),
        use_virtual_neighbors=cfg.get("use_virtual_neighbors", True),
        virtual_degree_threshold=int(cfg.get("virtual_degree_threshold", 5)),
        virtual_k=int(cfg.get("virtual_k", 5)),
        use_hard_negative_mining=cfg.get("use_hard_negative_mining", False),
        use_curriculum_learning=cfg.get("use_curriculum_learning", False),
        curriculum_warmup_epochs=int(cfg.get("curriculum_warmup_epochs", 100)),
        use_score_smoothing=cfg.get("use_score_smoothing", True),
        score_smoothing_alpha=float(cfg.get("score_smoothing_alpha", 0.3)),
        flow_t_sampling=cfg.get("flow_t_sampling", "logit_normal"),
        ensemble_score=cfg.get("ensemble_score", True),
        num_trial=args.num_trial if args.num_trial is not None else int(cfg.get("num_trial", 3)),
        exp_tag=cfg.get("exp_tag", None),
        flip_score=cfg.get("flip_score", False),
        kmeans_polarity=cfg.get("kmeans_polarity", False),
        kmeans_polarity_random_state=int(cfg.get("kmeans_polarity_random_state", args.seed)),
        kmeans_max_minority_ratio=float(cfg.get("kmeans_max_minority_ratio", 0.42)),
        polarity_hybrid=cfg.get("polarity_hybrid", True),
        quantile_rank_polarity=cfg.get("quantile_rank_polarity", False),
        quantile_rank_low=float(cfg.get("quantile_rank_low", 0.1)),
        quantile_rank_high=float(cfg.get("quantile_rank_high", 0.9)),
        quantile_rank_threshold=float(cfg.get("quantile_rank_threshold", 0.5)),
        lcc_spearman_polarity=cfg.get("lcc_spearman_polarity", False),
        lcc_spearman_threshold=float(cfg.get("lcc_spearman_threshold", -0.05)),
        lcc_polarity_mode=str(cfg.get("lcc_polarity_mode", "spearman")),
        lcc_tail_k_percent=float(cfg.get("lcc_tail_k_percent", 0.05)),
        lcc_tail_margin=float(cfg.get("lcc_tail_margin", 1.2)),
        iforest_anchor_polarity=cfg.get("iforest_anchor_polarity", False),
        iforest_n_estimators=int(cfg.get("iforest_n_estimators", 100)),
        iforest_anchor_random_state=int(cfg.get("iforest_anchor_random_state", args.seed)),
        smooth_discrepancy_polarity=cfg.get("smooth_discrepancy_polarity", False),
        smooth_discrepancy_mode=str(cfg.get("smooth_discrepancy_mode", "spearman")),
        smooth_discrepancy_spearman_threshold=float(cfg.get("smooth_discrepancy_spearman_threshold", -0.05)),
        smooth_discrepancy_k_percent=float(cfg.get("smooth_discrepancy_k_percent", 0.05)),
        smooth_discrepancy_tail_margin=float(cfg.get("smooth_discrepancy_tail_margin", 1.0)),
        smooth_discrepancy_representation=str(cfg.get("smooth_discrepancy_representation", "embedding")),
        smoothgnn_polarity=cfg.get("smoothgnn_polarity", False),
        smoothgnn_anchor_k_percent=float(cfg.get("smoothgnn_anchor_k_percent", 0.05)),
        smoothgnn_anchor_margin=float(cfg.get("smoothgnn_anchor_margin", 1.05)),
        smoothgnn_robust_spearman_threshold=float(cfg.get("smoothgnn_robust_spearman_threshold", -0.1)),
        smoothgnn_eps=float(cfg.get("smoothgnn_eps", 4e-3)),
        smoothgnn_full_polarity=cfg.get("smoothgnn_full_polarity", False),
        smoothgnn_full_nepoch=int(cfg.get("smoothgnn_full_nepoch", 100)),
        smoothgnn_full_hidden_dim=int(cfg.get("smoothgnn_full_hidden_dim", 64)),
        smoothgnn_full_spearman_threshold=float(cfg.get("smoothgnn_full_spearman_threshold", -0.05)),
        smoothgnn_full_seed=int(cfg.get("smoothgnn_full_seed", args.seed)),
        smoothgnn_teacher_polarity=cfg.get("smoothgnn_teacher_polarity", False),
        smoothgnn_teacher_dir=cfg.get("smoothgnn_teacher_dir", None),
        smoothgnn_teacher_spearman_threshold=float(cfg.get("smoothgnn_teacher_spearman_threshold", -0.1)),
    )

    print("Running FMGADself on dataset:", dset, "num_trial:", model.num_trial, flush=True)
    t0 = time.perf_counter()
    out = model(dset)
    elapsed = time.perf_counter() - t0
    print("FMGADself_TIME_SEC\t{:.1f}".format(elapsed), flush=True)
    if args.result_file:
        payload = {"dataset": dset, "seed": int(args.seed), "time_sec": elapsed, **out}
        if "auc_mean" in payload:
            payload["auc"] = float(payload["auc_mean"])
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()