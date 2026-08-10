#!/usr/bin/env bash
# Detached SmoothAdv training driver: 3 noise-augmented models, one per sigma.
# Launched with setsid+nohup so it survives Claude Code session teardown.
#
# SIGMAS is overridable so an interrupted run can retrain only what is missing:
#   SIGMAS="0.05 0.10" ./train_all.sh
cd /workspace/openkbp/open-kbp-modified || exit 1
export TRAINDATA=/tmp/okbp-data/provided-data
SIGMAS="${SIGMAS:-0.02 0.05 0.10}"

for S in $SIGMAS; do
  echo "=== START sigma=$S $(date -u +%H:%M:%S) ==="
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 \
      --ptv-weight 4.0 --no-jit --aug-noise "$S" --data-dir "$TRAINDATA" \
      --out-suffix "smoothadv_s$S" > "train_s$S.log" 2>&1
  echo "=== END sigma=$S exit=$? $(date -u +%H:%M:%S) ==="
  ls -d results/*smoothadv_s$S* 2>/dev/null
done

echo "=== ALL TRAINING DONE $(date -u +%H:%M:%S) ==="
find results -name "epoch_100.keras" 2>/dev/null
