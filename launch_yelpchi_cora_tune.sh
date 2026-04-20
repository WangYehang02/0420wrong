#!/usr/bin/env bash
# YelpChi + Cora 精细调参（detailed 网格随机子采样 × 多 seed），输出至 /mnt/yehang，结束后写入 ~/finalreport。
# 用法：nohup bash launch_yelpchi_cora_tune.sh > /mnt/yehang/yelpchi_cora_launcher.log 2>&1 &
set -euo pipefail

FMGAD_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="/home/yehang/miniconda3/etc/profile.d/conda.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/mnt/yehang/fmgad_tune_yelpchi_cora_${STAMP}"
mkdir -p "$OUT"

cd "$FMGAD_ROOT"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate fmgad

echo "[launcher] FMGAD_ROOT=$FMGAD_ROOT" | tee "$OUT/launcher_meta.log"
echo "[launcher] OUT=$OUT" | tee -a "$OUT/launcher_meta.log"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv >>"$OUT/launcher_meta.log" || true

read -r -a GPUS <<<"$(python pick_idle_gpus.py --want 8 --util-max 25 --mem-max-mib 15000)"
NW="${#GPUS[@]}"
if [[ "$NW" -lt 2 ]]; then
  GPUS=(0 1 2 3)
  NW=4
fi
echo "[launcher] Using GPUs: ${GPUS[*]} (workers=$NW)" | tee -a "$OUT/launcher_meta.log"

python run_tune_refined.py \
  --datasets yelpchi cora \
  --seeds 42 0 1 2 3 \
  --gpus "${GPUS[@]}" \
  --max-configs 36 \
  --search-mode detailed \
  --sampler-seed 20260420 \
  --num-trial 1 \
  --max-workers "$NW" \
  --timeout-sec 10800 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/console.log"

python write_tune_summary_to_finalreport.py --tune-dir "$OUT"
echo "[launcher] Done. Output: $OUT" | tee -a "$OUT/launcher_meta.log"
