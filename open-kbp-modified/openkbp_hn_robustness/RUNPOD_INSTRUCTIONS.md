# Robustness Pipeline — RunPod Instructions

## Overview

Run dose prediction inference on 40 validation patients × 11 conditions (1 baseline + 5 perturbations × 2 levels = 440 total predictions). Takes ~30-60 minutes on an RTX 3090/4090.

## Prerequisites

- RunPod pod with GPU (RTX 3090/4090 recommended)
- OpenKBP repo already cloned at `/workspace/open-kbp/`
- Trained model at `/workspace/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras`
- Validation data at `/workspace/open-kbp/provided-data/validation-pats/`
- TensorFlow 2.18.0 (must match the version used for training with mixed precision)

## Step-by-Step

### 1. Install dependencies (if not already installed)

```bash
pip install tensorflow[and-cuda]==2.18.0 pandas numpy scipy pyyaml tqdm more_itertools
```

### 2. Upload the robustness archive from your Mac

On your **local Mac**:
```bash
cd "/Users/neiltripathi/Library/Mobile Documents/com~apple~CloudDocs/Documents/OpenKBP Project"
runpodctl send robustness_runpod.tar.gz
```

On **RunPod**, receive the file:
```bash
cd /workspace/open-kbp
runpodctl receive <code_from_send>
```

### 3. Extract the archive

```bash
cd /workspace/open-kbp
tar xzf robustness_runpod.tar.gz
```

This creates `openkbp_hn_robustness/` with:
- `perturbations/` — perturbation module code
- `configs/default.yaml` — pipeline configuration
- `data_perturbed/P*/L*/pt_*/ct.csv` — 400 perturbed CT files (no symlinks yet)
- `run_inference.py` — the inference script
- `runpod_setup.sh` — symlink creation script

### 4. Create symlinks to validation data

The perturbed patient directories only contain `ct.csv`. They need symlinks to the original validation patient files (structure masks, dose, possible_dose_mask, voxel_dimensions) so the DataLoader can find them.

```bash
bash openkbp_hn_robustness/runpod_setup.sh /workspace/open-kbp/provided-data/validation-pats
```

Expected output:
```
Creating symlinks from perturbed patient dirs to validation data...
Done! Created symlinks for 400 patient directories.
```

### 5. Verify symlinks work (quick sanity check)

```bash
ls openkbp_hn_robustness/data_perturbed/P1_noise/L1/pt_201/
```

Should show `ct.csv` (real file) plus symlinks for all other files:
```
Brainstem.csv -> /workspace/open-kbp/provided-data/validation-pats/pt_201/Brainstem.csv
Esophagus.csv -> ...
ct.csv          (actual perturbed file)
dose.csv -> ...
...
```

### 6. Run inference

```bash
cd /workspace/open-kbp
python openkbp_hn_robustness/run_inference.py \
    --model /workspace/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras
```

This will:
1. Load the trained model
2. Run baseline inference on 40 unperturbed validation patients
3. Run inference on all 10 perturbed conditions (P1_noise/L1, P1_noise/L2, P2_bone_shift/L1, ..., P5_dental/L2)
4. Save 440 prediction CSVs to `openkbp_hn_robustness/predictions/`

The script **skips** patients that already have predictions, so it's safe to restart if interrupted.

If you want to run just a subset (e.g., to test first):
```bash
# Just baseline
python openkbp_hn_robustness/run_inference.py \
    --model /workspace/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras \
    --conditions baseline

# Just one perturbation
python openkbp_hn_robustness/run_inference.py \
    --model /workspace/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras \
    --conditions P1_noise/L1
```

### 7. Verify baseline matches known results

After baseline inference completes, you can quickly check the predictions look correct:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator
from provided_code.utils import get_paths

val_paths = sorted(Path('provided-data/validation-pats').glob('pt_*'))
pred_paths = sorted(get_paths(Path('openkbp_hn_robustness/predictions/baseline'), extension='csv'))
print(f'Reference patients: {len(val_paths)}, Predictions: {len(pred_paths)}')

ref = DataLoader(val_paths, normalize=False)
pred = DataLoader(pred_paths, normalize=False)
ev = DoseEvaluator(ref, pred)
ev.evaluate()
dose, dvh = ev.get_scores()
print(f'Dose Score: {dose:.3f} (expect ~3.73)')
print(f'DVH Score:  {dvh:.3f} (expect ~2.54)')
"
```

### 8. Package results for download

After all inference completes:

```bash
cd /workspace/open-kbp
tar czf /workspace/robustness_predictions.tar.gz openkbp_hn_robustness/predictions/
```

Then on **RunPod**:
```bash
runpodctl send /workspace/robustness_predictions.tar.gz
```

On your **local Mac**:
```bash
cd "/Users/neiltripathi/Library/Mobile Documents/com~apple~CloudDocs/Documents/OpenKBP Project/open-kbp-modified"
runpodctl receive <code>
tar xzf robustness_predictions.tar.gz
```

### 9. Run evaluation and visualization locally

Back on your Mac:
```bash
cd "/Users/neiltripathi/Library/Mobile Documents/com~apple~CloudDocs/Documents/OpenKBP Project/open-kbp-modified"

# Compute DVH and dose scores for all conditions
python openkbp_hn_robustness/evaluate_metrics.py

# Generate figures
python openkbp_hn_robustness/visualize_results.py
```

## Troubleshooting

**Model fails to load:** Make sure TensorFlow version matches training (2.18.0). Mixed precision models are not backward compatible.

**"No module named provided_code":** Make sure you're running from `/workspace/open-kbp/` (the directory containing `provided_code/`).

**"No module named openkbp_hn_robustness":** The script auto-adds its parent dir to sys.path. Make sure you extracted the tar in `/workspace/open-kbp/`.

**Out of memory:** The script uses `batch_size=1` and `cache_data=False`, so memory usage should be minimal. If still OOM, the model itself may be too large — try restarting the pod with a fresh GPU.

**Symlink errors:** If `runpod_setup.sh` reports missing patients, make sure validation data exists at the path you passed. Check with: `ls /workspace/open-kbp/provided-data/validation-pats/ | head`

## Expected Output Structure

```
openkbp_hn_robustness/predictions/
├── baseline/
│   ├── pt_201.csv
│   ├── pt_202.csv
│   └── ... (40 files)
├── P1_noise/
│   ├── L1/
│   │   ├── pt_201.csv
│   │   └── ... (40 files)
│   └── L2/
│       └── ... (40 files)
├── P2_bone_shift/
│   ├── L1/ ...
│   └── L2/ ...
├── P3_bias_field/
│   ├── L1/ ...
│   └── L2/ ...
├── P4_resolution/
│   ├── L1/ ...
│   └── L2/ ...
└── P5_dental/
    ├── L1/ ...
    └── L2/ ...
```

Total: 440 prediction CSV files (~11 conditions × 40 patients)
