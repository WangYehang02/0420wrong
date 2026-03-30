#!/usr/bin/env bash
# 八卡并行；固定 seed=42；每数据集最多 60 组超参 → 5×60=300 任务；输出在 /mnt/yehang；nohup 可脱离 Cursor。
set -euo pipefail
FMGAD_ROOT="/home/yehang/0330/FMGAD"
PY="/home/yehang/miniconda3/envs/fmgad/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="/mnt/yehang/fmgad_tune_detailed_seed42_max300_${STAMP}"
mkdir -p "$OUT"
cd "$FMGAD_ROOT"
nohup "$PY" run_tune_refined.py \
  --datasets weibo books reddit enron disney \
  --gpus 0 1 2 3 4 5 6 7 \
  --seeds 42 \
  --max-configs 60 \
  --sampler-seed 20260330 \
  --num-trial 1 \
  --max-workers 8 \
  --timeout-sec 14400 \
  --search-mode detailed \
  --output-dir "$OUT" \
  > "$OUT/nohup_tune.log" 2>&1 &
echo $! > "$OUT/pid.txt"
echo "PID $(cat "$OUT/pid.txt")"
echo "OUT $OUT"
echo "LOG $OUT/nohup_tune.log"
