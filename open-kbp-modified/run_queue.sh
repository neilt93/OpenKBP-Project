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

# /workspace is a MooseFS share with a ~20 GB QUOTA -- `df` reports the whole
# cluster (1.2 PB free) and is useless here. Exceeding it truncates the file
# being written and kills the writer (EDQUOT), which is what ended the 00:45 run
# mid-checkpoint. runpod_train.py writes a 1.79 GB checkpoint every 10 epochs and
# keeps the last 5, so ONE training run can add ~9 GB. Prune before each train.
prune_ckpts() {
  find /workspace/results -name "epoch_*.keras" ! -name "epoch_100.keras" -delete 2>/dev/null
  echo "--- pruned intermediate checkpoints; /workspace/results now $(du -sh /workspace/results 2>/dev/null | cut -f1) ---"
}

radii_for() {
  case "$1" in
    0.02) echo "0.01,0.02,0.028" ;;
    0.05) echo "0.025,0.05,0.07" ;;
    0.10) echo "0.05,0.1,0.14" ;;
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
  # RESUMABLE: never recompute a finished certificate. The seed-replicate
  # training was killed mid-checkpoint-write at 00:45 (EDQUOT -- /workspace hit
  # its ~20 GB quota) and took the queue down with it. Without this guard a
  # restart would redo ~105 min of control certification that already succeeded.
  if [ -f "$OUT/certify_summary.json" ]; then
    banner "SKIP $OUT (certificate already present)"
    return 0
  fi
  local ok=0
  # 32 OOMs on the 4090 (24 GB); 16 runs at ~52 s/patient. Starting at 32 would
  # burn a guaranteed-failed attempt on each of the 5 certify calls below.
  for BD in 16 8; do
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
# Wait on the DRIVER scripts too, not just the python workers. certify_all.sh
# runs one certify_smoothing.py per sigma, so between sigma=0.05 finishing and
# sigma=0.10 starting there is a window with no python process alive. A 60 s
# poll landing in that window would start control training on top of the next
# certification and contend for (or OOM) the GPU.
gpu_busy() {
  pgrep -f "runpod_train.py"        >/dev/null 2>&1 && return 0
  pgrep -f "certify_smoothing.py"   >/dev/null 2>&1 && return 0
  pgrep -f "bash ./certify_all.sh"  >/dev/null 2>&1 && return 0
  pgrep -f "bash ./train_all.sh"    >/dev/null 2>&1 && return 0
  return 1
}
while gpu_busy; do sleep 60; done
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

# --- 3. SEED REPLICATE at sigma=0.05: the missing noise floor -----------------
# Every arm is a single training run, so we currently cannot say whether a -12%
# D95 shift is the method or just run-to-run variance. A second independently
# trained sigma=0.05 model, identical config, differing only in seed, measures
# that variance directly: the gap between the two IS the noise floor that every
# other delta has to clear. Placed before the n=500 steps deliberately -- it
# removes a confound, whereas n=500 only extends the radius grid, so if the pod
# dies early this is the one worth having.
SEED_SUFFIX=smoothadv_s0.05_seed1
if find_model "$SEED_SUFFIX" >/dev/null 2>&1; then
  banner "seed replicate already present, skipping training"
else
  prune_ckpts
  banner "TRAIN seed replicate (sigma=0.05, --seed 1)"
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 \
      --ptv-weight 4.0 --no-jit --aug-noise 0.05 --seed 1 --data-dir "$TRAINDATA" \
      --out-suffix "$SEED_SUFFIX" > train_seed1.log 2>&1
  if [ $? -ne 0 ]; then fail "seed-replicate training crashed (see train_seed1.log)"; fi
fi

SEEDM=$(find_model "$SEED_SUFFIX")
if [ -n "$SEEDM" ]; then
  banner "seed replicate = $SEEDM"
  certify "$SEEDM" 0.05 "$(radii_for 0.05)" 100 - "certify_seed1_s0.05"
else
  fail "no seed-replicate model after training -- noise floor unmeasured"
fi

# --- 4-5. n=500 tightening at sigma=0.02 on 10 pts (larger radius grid) ---
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
ls -la certify_control_s*/certify_summary.json certify_seed1_s0.05/certify_summary.json \
       certify_n500_s0.02_*/certify_summary.json 2>&1
if [ "$failed" -ne 0 ]; then
  echo "!!! QUEUE FINISHED WITH FAILURES -- inspect the *.log files above."
  exit 1
fi
echo "=== ALL QUEUE STEPS OK ==="
