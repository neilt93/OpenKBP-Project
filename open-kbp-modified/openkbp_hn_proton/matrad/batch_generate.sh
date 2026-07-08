#!/usr/bin/env bash
# Phase 2 — batch-generate proton IMPT ground-truth dose for a range of OpenKBP
# patients, then assemble an OpenKBP-format proton training dir (CT + structure
# masks taken from OpenKBP, dose.csv replaced by the generated proton dose).
#
# matRad dose calc is CPU-only and x86 — run this on a NATIVE x86_64 box (RunPod
# CPU instance), NOT under Apple-Silicon emulation, for the full 3 mm run. No GPU
# is used in Phase 2; the GPU is only for Phase 3 (training the U-Net).
#
# Usage (run from open-kbp-modified/):
#   openkbp_hn_proton/matrad/batch_generate.sh <data_root> <start> <end> <proton_data_out>
#     data_root         dir containing pt_<n>/  (e.g. provided-data/train-pats)
#     start end         inclusive pt_<n> range  (e.g. 1 200)
#     proton_data_out   where to assemble the proton training split
#                       (e.g. proton-data/train-pats)
#
# Env:
#   MATRAD_DIR   matRad checkout (default: SanDisk warehouse; on a box, point at the
#                on-box clone). IMAGE   docker tag (default openkbp-matrad:octave640).
#   JOBS_NOTE    parallelism is by RANGE, not inside this script — run several
#                invocations on DISJOINT ranges, each with its OWN MATRAD_DIR copy
#                (docker_octave mounts matRad read-write, so copies must not be shared).
#
# Re-runnable: build/matRad/import each skip patients already done, so a crashed run
# resumes by re-invoking the same command.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # openkbp_hn_proton/
CASES_DIR="$MODULE_DIR/matrad_cases"
PROTON_DOSE="$MODULE_DIR/proton_dose"

DATA_ROOT="${1:?usage: batch_generate.sh <data_root> <start> <end> <proton_data_out>}"
START="${2:?start}"
END="${3:?end}"
PROTON_DATA_OUT="${4:?proton_data_out}"

echo "== Phase 2 batch: pt_${START}..pt_${END} from ${DATA_ROOT} =="

# 1. Build matRad inputs for the range (build_case skips missing patients).
python "$MODULE_DIR/build_case.py" --patient-range "$START" "$END" --data-root "$DATA_ROOT" --no-qc

# 2. Optimise IMPT plans for every *_input.mat (run_plan.m skips cases with a dose).
#    This is the heavy, CPU-bound step. For 240 patients split the range across
#    parallel invocations (see JOBS_NOTE above); each container runs sequentially.
"$MODULE_DIR/matrad/docker_octave.sh" "$CASES_DIR"

# 3. Import each dose cube to an OpenKBP sparse dose.csv, and assemble the training
#    dir: symlink every OpenKBP file EXCEPT dose.csv, then drop in the proton dose.
for n in $(seq "$START" "$END"); do
    pid="pt_${n}"
    src="$DATA_ROOT/$pid"
    dose_mat="$CASES_DIR/${pid}_dose.mat"
    [ -d "$src" ] || continue
    if [ ! -f "$dose_mat" ]; then
        echo "  WARN $pid: no ${pid}_dose.mat (matRad did not finish it) — skipping"
        continue
    fi
    proton_csv="$PROTON_DOSE/$pid/dose.csv"
    if [ ! -f "$proton_csv" ]; then
        python "$MODULE_DIR/import_dose.py" --result "$dose_mat" --out "$proton_csv"
    fi
    # Assemble OpenKBP-format proton patient dir.
    dst="$PROTON_DATA_OUT/$pid"
    mkdir -p "$dst"
    for f in "$src"/*; do
        base="$(basename "$f")"
        [ "$base" = "dose.csv" ] && continue          # proton dose replaces photon dose
        ln -sf "$(cd "$(dirname "$f")" && pwd)/$base" "$dst/$base"
    done
    cp -f "$proton_csv" "$dst/dose.csv"
done

echo "== Done pt_${START}..pt_${END}. Proton training dir: ${PROTON_DATA_OUT} =="
echo "   (each patient = OpenKBP CT+masks symlinked + generated proton dose.csv)"
