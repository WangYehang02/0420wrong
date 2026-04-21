# 已移除的无监督极性探针（归档说明）

主仓库现已**仅保留**与 `FMGAD_五数据集_seeds42_0_1_2_3.md` 一致的主线：**基于局部聚类系数（LCC）的极性校准**（`lcc_polarity_mode: spearman` 或 `tail`），实现见 `utils.py` 中 `calibrate_polarity_lcc_spearman`、`calibrate_polarity_tail_lcc` 与 `compute_node_lcc_tensor`，`res_flow_gad.sample()` 内优先走该分支。

下列方案**已从代码中删除**（不再维护）；历史实验的 Markdown 与 JSON 已移至 **`/home/yehang/finalreport/archive_fmgad_alternate_probes/`**（见该目录下 `README.txt`）。

## 1. 特征锚点（Feature Anchor）

- **思路**：用 `data.x` 维中位数作「正常原型」，对分数最高/最低各约 k% 节点算余弦相似度，按差值判定是否翻转。
- **曾用文件**：`utils.calibrate_polarity_feature_anchor`（已删）。
- **报告示例**（已归档）：`archive_fmgad_alternate_probes/FMGAD_feature_anchor_五数据集五seed.md`，子目录 `fmgad_feature_anchor_5seeds_42_0_1_2_3/`、`fmgad_feature_anchor_seed42/`。

## 2. 孤立森林基线锚定（IForest + Spearman）

- **思路**：在 `data.x` 上拟合一次 IsolationForest，用 `-decision_function` 作特征基线分；在 `sample()` 时间步循环内只做与模型分的 Spearman 相关与可选翻转。
- **曾用文件**：`compute_iforest_baseline_scores`、`calibrate_polarity_iforest_precomputed`（已删）。
- **报告示例**（已归档）：`archive_fmgad_alternate_probes/FMGAD_iforest_anchor_五数据集五seed.md`，子目录 `fmgad_iforest_anchor_5seeds_42_0_1_2_3/`。

## 3. 其它独立实验报告（非当前主线代码）

- **Tail-LCC 专项汇总**（已归档）：`FMGAD_五数据集_TailLCC_seeds42_0_1_2_3.md` 与 `fmgad_tail_lcc_seeds42_0_1_2_3/`（与主线同为 LCC 探针，仅实验记录单独归档）。
- **其它**（已归档）：`极端5seed.md`、`FMGAD_YelpChi_Cora_tune_best_*.md/json` 等。

恢复旧探针需从本归档目录对照历史 JSON 与 git 历史自行还原，本仓库默认不再包含相关实现。
