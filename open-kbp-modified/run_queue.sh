#!/usr/bin/env bash
# ============================================================================
# 5-hour experiment queue for the SmoothAdv certified-robustness study.
#
# Run this AFTER (or alongside) the current certify pass. It waits for any
# running certify/train job to finish, then executes, in value order:
#
#   1. Train a MATCHED CONTROL model: same numpy-augmentation path as the
#      SmoothAdv arms, but with ~zero noise (--aug-noise 1e-9). This is the
#      only way to force the numpy aug path at zero noise -- --aug-noise 0
#      falls back to the lightweight TF path, which is the very confound we
#      are removing. 1e-9 normalized ~= 4e-6 HU = physically zero.
#   2. Certify the control at sigma = 0.02, 0.05, 0.10 (n=100, 40 pts) on the
#      SAME radii grids as the committed baselines -> clean "noise-training
#      vs not" comparison at every arm.
#   3-4. n=500 tightening at sigma=0.02 on 10 patients (SmoothAdv + control):
#      n=500 lifts R_max from 1.45*sigma to 2.17*sigma -> a larger certified
#      radius for the headline, and the tighter comparison stays clean.
#
# FAIL LOUD: any missing model or failed step prints a screaming banner and
# sets a non-zero exit. Never a silent skip.
#
# Launch detached:
#   cd /workspace/openkbp/open-kbp-modified
#   nohup bash ./run_queue.sh > run_queue.log 2>&1 &
#   tail -f run_queue.log
# ============================================================================
set -u
cd /workspace/openkbp/open-kbp-modified || exit 1

DATA=/tmp/okbp-data/provided-data/validation-pats
TRAINDATA=/tmp/okbp-data/provided-data
CTRL_SUFFIX=control_augpath
failed=0

banner()  { echo; echo "=== $* $(date -u +%H:%M:%S) ==="; }
fail()    { echo "!!! FAILURE: $*"; failed=1; }

radii_for() {
  case "$1" in
    0.02) echo "0.01,0.02,0.028,0.5,1.0,2.0" ;;
    0.05) echo "0.025,0.05,0.07,0.5,1.0,2.0" ;;
    0.10) echo "0.05,0.1,0.14,0.5,1.0,2.0" ;;
  esac
}

find_model() {  # find_model <substring> ; echoes first epoch_100.keras match
  local pat="$1" m
  for base in /workspace/results ./results ../results; do
    m=$(ls -d "$base"/*"$pat"*/models/epoch_100.keras 2>/dev/null | head -1)
    [ -n "$m" ] && { echo "$m"; return 0; }
  done
  return 1
}

certify() {  # certify <model> <sigma> <radii> <n_samples> <n_patients|-> <outdir>
  local M="$1" S="$2" R="$3" N="$4" NP="$5" OUT="$6" np_arg=""
  [ "$NP" != "-" ] && np_arg="--n-patients $NP"
  local ok=0
  for BD in 32 16 8; do
    echo "--- certify sigma=$S n=$N bd=$BD -> $OUT ---"
    python certify_smoothing.py --model "$M" --data-dir "$DATA" --sigma "$S" \
        --n-samples "$N" --batch-draws "$BD" --radii "$R" --tol-gy 1.0 --alpha 0.001 \
        $np_arg --output "$OUT/" > "${OUT}.log" 2>&1
    if [ $? -eq 0 ] && [ -f "$OUT/certify_summary.json" ]; then
      banner "DONE certify $OUT (bd=$BD)"; ok=1; break
    fi
    echo "--- bd=$BD failed, tail: ---"; tail -5 "${OUT}.log"
  done
  [ "$ok" -eq 1 ] || fail "certify $OUT failed at every batch-draws"
}

# --- 0. wait for any running train/certify job so we don't fight for the GPU ---
banner "queue start; waiting for any running train/certify to finish"
while pgrep -f "runpod_train.py" >/dev/null 2>&1 || pgrep -f "certify_smoothing.py" >/dev/null 2>&1; do
  sleep 60
done
banner "GPU free; starting queue"
nvidia-smi 2>/dev/null | head -15

# --- 1. train matched control (numpy aug path, ~zero noise) ---
if find_model "$CTRL_SUFFIX" >/dev/null 2>&1; then
  banner "control model already present, skipping training"
else
  banner "TRAIN matched control (--aug-noise 1e-9, suffix $CTRL_SUFFIX)"
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 \
      --ptv-weight 4.0 --no-jit --aug-noise 1e-9 --data-dir "$TRAINDATA" \
      --out-suffix "$CTRL_SUFFIX" > train_control.log 2>&1
  if [ $? -ne 0 ]; then fail "control training crashed (see train_control.log)"; fi
fi

CTRL=$(find_model "$CTRL_SUFFIX")
if [ -z "$CTRL" ]; then
  fail "no control model after training -- cannot run steps 2-4"
  echo "!!! QUEUE ABORTED"; exit 1
fi
banner "control model = $CTRL"

# --- 2. certify control at each sigma (n=100, 40 pts) ---
for S in 0.02 0.05 0.10; do
  certify "$CTRL" "$S" "$(radii_for $S)" 100 - "certify_control_s$S"
done

# --- 3-4. n=500 tightening at sigma=0.02 on 10 pts (larger radius grid) ---
N500_RADII="0.01,0.02,0.028,0.043"   # up to 2.17*sigma at n=500
SADV=$(find_model "smoothadv_s0.02")
if [ -n "$SADV" ]; then
  certify "$SADV" 0.02 "$N500_RADII" 500 10 "certify_n500_s0.02_smoothadv"
else
  fail "no smoothadv_s0.02 model found for n=500 tightening"
fi
certify "$CTRL" 0.02 "$N500_RADII" 500 10 "certify_n500_s0.02_control"

# --- summary ---
banner "QUEUE COMPLETE"
echo "certificates produced:"
ls -la certify_control_s*/certify_summary.json certify_n500_s0.02_*/certify_summary.json 2>&1
if [ "$failed" -ne 0 ]; then
  echo "!!! QUEUE FINISHED WITH FAILURES -- inspect the *.log files above."
  exit 1
fi
echo "=== ALL QUEUE STEPS OK ==="
