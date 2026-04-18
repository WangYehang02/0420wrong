# FMGAD 五数据集 × 五 Seed 结果汇总

**Seeds：** 1，12，123，42，66  

**说明：** 下列指标来自 `configs/*_best.yaml`，`flip_score` 为各数据集拓扑先验配置；极性仅在 `sample()` 中翻转一次（已修复 `ensemble` 二次翻转问题）。

**数据来源（可复现）：**

| 范围 | 日志目录 |
|------|----------|
| weibo / reddit / books / enron | `/home/yehang/comp/0417/logs/flip_single_20260418_123007/` |
| disney（`flip_score: true` 后单独复跑） | `/home/yehang/comp/0417/logs/disney_flip_true_20260418_133353/` |

Disney 在全五数据集联合跑批时曾为 `flip_score: false`；随后在 `configs/disney_best.yaml` 中改为 `flip_score: true` 并单独重跑，故下表中 **Disney 行与另外四数据集不在同一次 nohup 任务中**，但 **seeds、环境与主代码版本一致**。

---

## 汇总表（AUC）

| 数据集 | AUC mean ± std | 最佳 AUC | 最佳 seed |
|--------|----------------|----------|-----------|
| weibo | 0.9423 ± 0.0004 | 0.9428 | 42 |
| enron | 0.8463 ± 0.0324 | 0.8934 | 12 |
| disney | 0.7613 ± 0.0647 | 0.8418 | 123 |
| books | 0.6287 ± 0.0174 | 0.6576 | 66 |
| reddit | 0.5646 ± 0.0172 | 0.5927 | 42 |

---

## 分数据集明细（AUC / AP）

### weibo

| seed | AUC | AP |
|------|-----|-----|
| 1 | 0.9421 | 0.3436 |
| 12 | 0.9427 | 0.3438 |
| 123 | 0.9421 | 0.3435 |
| 42 | 0.9428 | 0.3434 |
| 66 | 0.9420 | 0.3436 |

### enron

| seed | AUC | AP |
|------|-----|-----|
| 1 | 0.8020 | 0.0013 |
| 12 | 0.8934 | 0.0029 |
| 123 | 0.8427 | 0.0027 |
| 42 | 0.8440 | 0.0098 |
| 66 | 0.8493 | 0.0014 |

### disney（`flip_score: true`）

| seed | AUC | AP |
|------|-----|-----|
| 1 | 0.6737 | 0.1017 |
| 12 | 0.7910 | 0.1519 |
| 123 | 0.8418 | 0.4577 |
| 42 | 0.7232 | 0.0990 |
| 66 | 0.7768 | 0.1271 |

### books

| seed | AUC | AP |
|------|-----|-----|
| 1 | 0.6162 | 0.0265 |
| 12 | 0.6148 | 0.0261 |
| 123 | 0.6237 | 0.0304 |
| 42 | 0.6314 | 0.0300 |
| 66 | 0.6576 | 0.0407 |

### reddit

| seed | AUC | AP |
|------|-----|-----|
| 1 | 0.5462 | 0.0353 |
| 12 | 0.5653 | 0.0370 |
| 123 | 0.5597 | 0.0373 |
| 42 | 0.5927 | 0.0427 |
| 66 | 0.5592 | 0.0368 |

---

## 一键复现（参考）

```bash
conda activate fmgad
cd /home/yehang/comp/0417
python run_best_eval.py \
  --datasets weibo reddit disney books enron \
  --seeds 1 12 123 42 66 \
  --gpus 0 1 2 3 4 \
  --output-dir <你的输出目录> \
  --report <你的输出目录>/report.md
```

（需保证当前 `configs/disney_best.yaml` 中已为 `flip_score: true`，与其它数据集 YAML 一致。）
