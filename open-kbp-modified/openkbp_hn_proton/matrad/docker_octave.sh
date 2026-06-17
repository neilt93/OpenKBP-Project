#!/usr/bin/env bash
# Run matRad's run_plan.m inside the openkbp-matrad Octave container.
#
# Usage:
#   docker_octave.sh <cases_dir> [output_dir]
#
#   <cases_dir>   dir containing <pid>_input.mat (from build_case.py). Output
#                 <pid>_dose.mat is written here (or to output_dir if given).
#
# Env:
#   MATRAD_DIR    path to a matRad checkout (default: SanDisk warehouse clone)
#   IMAGE         docker image tag (default: openkbp-matrad:octave640)
#
# matRad is mounted read-write because matRad_rc may touch its own folders; the
# image stays small (matRad is not baked in). amd64 is forced so the precompiled
# ipopt.mexoct640a64 solver loads (under emulation on Apple Silicon).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="${1:?usage: docker_octave.sh <cases_dir> [output_dir]}"
OUT_DIR="${2:-$CASES_DIR}"
IMAGE="${IMAGE:-openkbp-matrad:octave640}"
# NB: plain double-quoted assignment so the apostrophe in the volume name is
# literal — the ${VAR:-default} form mis-parses a "'" inside the default word.
if [ -z "${MATRAD_DIR:-}" ]; then
    MATRAD_DIR="/Volumes/Neil's SanDisk/OpenKBP-Warehouse/tools/matRad"
fi

CASES_DIR="$(cd "$CASES_DIR" && pwd)"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

[ -d "$MATRAD_DIR" ]   || { echo "matRad not found at: $MATRAD_DIR  (set MATRAD_DIR)"; exit 1; }
[ -f "$MATRAD_DIR/matRad_rc.m" ] || { echo "matRad_rc.m missing in $MATRAD_DIR — bad matRad checkout?"; exit 1; }

echo "matRad : $MATRAD_DIR"
echo "cases  : $CASES_DIR"
echo "output : $OUT_DIR"
echo "image  : $IMAGE"

# Mount an Octave entrypoint (run_entry.m) instead of an inline --eval string, to
# avoid fragile nested shell/Octave quoting. -w sets cwd so matRad_rc is found.
exec docker run --rm --platform linux/amd64 \
    -v "$MATRAD_DIR":/opt/matRad \
    -v "$SCRIPT_DIR":/work/scripts:ro \
    -v "$CASES_DIR":/work/cases \
    -v "$OUT_DIR":/work/out \
    -w /opt/matRad \
    "$IMAGE" \
    octave --no-gui /work/scripts/run_entry.m
