#!/bin/bash
# RunPod setup for robustness pipeline
# Run from /workspace after extracting robustness_runpod.tar.gz
#
# Prerequisites:
#   - OpenKBP repo cloned at /workspace/open-kbp/ (or wherever your data lives)
#   - Trained model available
#   - TensorFlow 2.18.0 installed
#
# Usage:
#   cd /workspace
#   tar xzf robustness_runpod.tar.gz
#   bash openkbp_hn_robustness/runpod_setup.sh /workspace/open-kbp/provided-data/validation-pats
#   cd open-kbp  # or wherever your open-kbp-modified code lives
#   python openkbp_hn_robustness/run_inference.py --model <path_to_model>

set -e

VALIDATION_DIR="${1:?Usage: $0 <path-to-validation-pats>}"

if [ ! -d "$VALIDATION_DIR" ]; then
    echo "ERROR: Validation directory not found: $VALIDATION_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PERTURBED="$SCRIPT_DIR/data_perturbed"

echo "Creating symlinks from perturbed patient dirs to validation data..."
echo "  Validation dir: $VALIDATION_DIR"
echo "  Perturbed root: $DATA_PERTURBED"

count=0
for perturbation_dir in "$DATA_PERTURBED"/P*; do
    [ -d "$perturbation_dir" ] || continue
    for level_dir in "$perturbation_dir"/L*; do
        [ -d "$level_dir" ] || continue
        for patient_dir in "$level_dir"/pt_*; do
            [ -d "$patient_dir" ] || continue
            patient_id=$(basename "$patient_dir")
            src_patient="$VALIDATION_DIR/$patient_id"

            if [ ! -d "$src_patient" ]; then
                echo "WARNING: $src_patient not found, skipping"
                continue
            fi

            # Symlink all files except ct.csv (which is the perturbed version)
            for src_file in "$src_patient"/*; do
                fname=$(basename "$src_file")
                if [ "$fname" = "ct.csv" ]; then
                    continue
                fi
                dst_file="$patient_dir/$fname"
                if [ ! -e "$dst_file" ]; then
                    ln -s "$src_file" "$dst_file"
                fi
            done
            count=$((count + 1))
        done
    done
done

echo "Done! Created symlinks for $count patient directories."
echo ""
echo "Next step: run inference"
echo "  cd <your-open-kbp-modified-dir>"
echo "  python openkbp_hn_robustness/run_inference.py --model <path_to_model>"
