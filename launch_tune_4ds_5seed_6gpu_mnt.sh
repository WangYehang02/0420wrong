#!/usr/bin/env bash
# 精细调参：weibo/books/reddit/enron（不含 Disney）；5 个 seed；GPU 2–7 六卡并行；输出 /mnt/yehang。
set -euo pipefail
FMGAD_ROOT="/home/yehang/0330/FMGAD"
PY="/home/yehang/miniconda3/envs/fmgad/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="/mnt/yehang/fmgad_tune_4ds_5seed_gpu234567_${STAMP}"
mkdir -p "$OUT"
cd "$FMGAD_ROOT"
nohup "$PY" run_tune_refined.py \
  --datasets weibo books reddit enron \
  --gpus 2 3 4 5 6 7 \
  --seeds 42 66 123 256 512 \
  --max-configs 108 \
  --sampler-seed 20260330 \
  --num-trial 1 \
  --max-workers 6 \
  --timeout-sec 14400 \
  --search-mode detailed \
  --output-dir "$OUT" \
  > "$OUT/nohup_tune.log" 2>&1 &
echo $! > "$OUT/pid.txt"
echo "PID $(cat "$OUT/pid.txt")"
echo "OUT $OUT"
echo "TASKS=$((4 * 108 * 5))"
