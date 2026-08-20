#!/usr/bin/env bash
# Detached certification driver. Waits for train_all.sh to finish, then re-certifies
# each noise-trained model at its matching sigma, on the informative radii grid
# (0.5x/1.0x/1.4x sigma). We pass --radii explicitly because certify_smoothing.py's
# default (0.5/1/2) is vacuous at every sane sigma. Those vacuous points are no longer
# appended here -- certify_smoothing.py now drops any radius above R_max anyway.
#
# MODEL PATH: runpod_train.py writes to the ABSOLUTE path /workspace/results when
# /workspace exists (RunPod special-case, runpod_train.py:147) -- NOT ./results as
# the brief states. Globbing the relative path finds nothing.
#
# FAIL LOUD: a missing model or a failed certification must be a screaming,
# non-zero exit -- never a silent skip that yields an empty result set hours later.
#
# --batch-draws is a pure GPU-batching knob: draw_smoothed_samples consumes
# rng.normal() sequentially, so grouping 100 draws as 32+32+32+4 vs 8x12+4 yields
# identical values in identical order. We try large first and fall back on OOM.
cd /workspace/openkbp/open-kbp-modified || exit 1
DATA=/tmp/okbp-data/provided-data/validation-pats
SIGMAS="${SIGMAS:-0.02 0.05 0.10}"
n_want=$(echo $SIGMAS | wc -w)
failed=0

# wait for training to finish (match the script invocation, not any stray string)
while pgrep -f "bash ./train_all.sh" > /dev/null 2>&1; do sleep 60; done
echo "=== training finished, starting certification $(date -u +%H:%M:%S) ==="

radii_for() {
  case "$1" in
    0.02) echo "0.01,0.02,0.028" ;;
    0.05) echo "0.025,0.05,0.07" ;;
    0.10) echo "0.05,0.1,0.14" ;;
  esac
}

find_model() {
  local S="$1" m
  for base in /workspace/results ./results ../results; do
    m=$(ls -d "$base"/*smoothadv_s"$S"*/models/epoch_100.keras 2>/dev/null | head -1)
    [ -n "$m" ] && { echo "$m"; return 0; }
  done
  return 1
}

for S in $SIGMAS; do
  M=$(find_model "$S")
  if [ -z "$M" ]; then
    echo "!!! FAILURE sigma=$S: NO MODEL FOUND (searched /workspace/results, ./results, ../results)"
    echo "!!! training for this sigma did not produce models/epoch_100.keras -- check train_s$S.log"
    failed=1
    continue
  fi
  R=$(radii_for "$S")
  echo "=== CERTIFY sigma=$S model=$M radii=$R $(date -u +%H:%M:%S) ==="

  ok=0
  # 32 is known to OOM on the 4090 (24 GB); 16 runs at ~61 s/patient. Start at 16
  # rather than burning a guaranteed-failed attempt per sigma.
  for BD in 16 8; do
    echo "--- attempting --batch-draws $BD ---"
    python certify_smoothing.py --model "$M" --data-dir "$DATA" --sigma "$S" \
        --n-samples 100 --batch-draws "$BD" --radii "$R" --tol-gy 1.0 --alpha 0.001 \
        --output "certify_smoothadv_s$S/" > "certify_s$S.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "certify_smoothadv_s$S/certify_summary.json" ]; then
      echo "=== DONE sigma=$S (batch-draws $BD) $(date -u +%H:%M:%S) ==="
      ok=1
      break
    fi
    echo "--- batch-draws $BD failed (rc=$rc), tail: ---"
    tail -5 "certify_s$S.log"
  done

  if [ "$ok" -ne 1 ]; then
    echo "!!! FAILURE sigma=$S: certification failed at every --batch-draws (32/16/8)"
    echo "!!! see certify_s$S.log"
    failed=1
  fi
done

echo "=== CERTIFICATION PASS COMPLETE $(date -u +%H:%M:%S) ==="
# count only the sigmas THIS run asked for -- a stale directory from an earlier
# run must not be able to make an incomplete pass look complete.
n=0
for S in $SIGMAS; do
  [ -f "certify_smoothadv_s$S/certify_summary.json" ] && n=$((n + 1))
done
echo "certificates produced: $n / $n_want"
ls -la certify_smoothadv_s*/certify_summary.json 2>&1

if [ "$failed" -ne 0 ] || [ "$n" -lt "$n_want" ]; then
  echo "!!! CERTIFICATION INCOMPLETE -- $n/$n_want produced. DO NOT treat this run as a result."
  exit 1
fi
echo "=== ALL CERTIFICATION DONE ($n/$n_want) ==="
