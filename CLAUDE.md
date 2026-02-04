# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# OpenKBP Dose Prediction Project

Radiotherapy dose prediction using 3D U-Net for head-and-neck cancer patients.

## Best Results

| Model | DVH Score | Dose Score | Config |
|-------|-----------|------------|--------|
| **Best (Iteration 2)** | 2.535 | 3.731 | 64 filters, 100 epochs, SE blocks, augmentation, PTV weight 4.0, CT_MAX=4095 |
| Iteration 1 | 2.563 | 3.856 | Same config, CT_MAX=3000 |
| Baseline | 11.481 | 7.180 | 64 filters, 100 epochs, original architecture |
| Competition Winner | 1.478 | 2.429 | Ensemble + cascade |

## Project Structure

```
OpenKBP Project/
├── open-kbp-modified/     # Main code
│   ├── runpod_train.py    # Training script with CLI
│   ├── provided_code/     # Core modules
│   │   ├── network_architectures.py  # U-Net with SE blocks
│   │   ├── network_functions.py      # Training logic
│   │   ├── data_loader.py            # Data loading with caching
│   │   └── dose_evaluation_class.py  # DVH/Dose scoring
│   └── provided-data/     # Training data (200 patients)
└── results/               # Trained models and predictions
```

## Training Commands

### Best Config (RunPod)
```bash
python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 --ptv-weight 4.0 --no-jit
```

### Resume Training to 200 Epochs
```bash
python runpod_train.py --filters 64 --epochs 200 --use-se --use-aug --batch-size 4 --ptv-weight 4.0 --no-jit
```

### Ensemble Training (5 seeds)
```bash
for seed in 1 2 3 4 5; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 --ptv-weight 4.0 --no-jit --seed $seed
done
```

## Key Improvements Over Baseline

1. **InstanceNormalization** - Better than BatchNorm for small batches
2. **SE Blocks** - Channel attention on deep layers (x4+)
3. **Residual Connections** - Encoder and decoder blocks
4. **Masked MAE Loss** - Only compute loss in possible_dose_mask region
5. **PTV Weighting** - 5x weight on target structures (ptv_weight=4.0)
6. **Data Augmentation** - LR/AP flips, CT intensity scaling
7. **Mixed Precision** - float16 training for 2x speedup
8. **Data Caching** - Pre-stack all patients for instant batch slicing

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--filters` | 64 | Initial U-Net filters |
| `--epochs` | 100 | Training epochs |
| `--use-se` | off | Enable SE attention blocks |
| `--use-aug` | off | Enable data augmentation |
| `--ptv-weight` | 2.0 | Extra weight on PTV voxels |
| `--batch-size` | 2 | Batch size |
| `--no-jit` | off | Disable XLA JIT (needed for RunPod) |
| `--seed` | None | Random seed for ensembling |

## RunPod Setup

See `open-kbp-modified/RUNPOD_SETUP.md` for full instructions.

Quick start:
```bash
cd /workspace && git clone https://github.com/ababier/open-kbp.git && cd open-kbp
pip install tensorflow[and-cuda]==2.18.0 pandas numpy scipy tqdm more_itertools
# Upload code-update.zip via runpodctl
python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 --ptv-weight 4.0 --no-jit
```

## Data Format

- **CT**: 128x128x128 HU values, normalized to [0,1] (clip 0-4095 per official docs)
- **Dose**: 128x128x128 Gy values, normalized by 70 Gy prescription
- **Structures**: 10 ROI masks (Brainstem, SpinalCord, RightParotid, LeftParotid, Esophagus, Larynx, Mandible, PTV56, PTV63, PTV70)

## Evaluation Metrics

- **DVH Score**: Mean absolute error of DVH metrics (D_1, D_95, D_99, mean) across all structures
- **Dose Score**: Voxel-wise MAE within possible_dose_mask region

## Adversarial Robustness Evaluation

Test model robustness using FGSM and PGD attacks on CT inputs.

### Commands (Run on RunPod with TF 2.18.0)

```bash
# Evaluate adversarial robustness
python adversarial_eval.py \
    --model results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras \
    --attack fgsm pgd \
    --epsilons 0,0.001,0.005,0.01,0.02,0.05,0.1 \
    --output adversarial_results/

# Generate plots (histograms, box plots, degradation charts)
python plot_adversarial.py --results-dir adversarial_results/
```

### Attack Details

- **FGSM** (Fast Gradient Sign Method): Single-step attack using sign of gradient
- **PGD** (Projected Gradient Descent): Multi-step iterative attack (10 steps default)
- **Epsilon**: Perturbation magnitude in normalized CT space [0,1]. Convert to HU: `HU = epsilon * 4095`
  - ε=0.01 ≈ 41 HU (within typical CT noise of 10-50 HU)
  - ε=0.05 ≈ 205 HU
  - ε=0.1 ≈ 410 HU
- Attacks maximize MAE loss by perturbing CT inputs while keeping structure masks unchanged

### Important Notes

**TensorFlow Version Compatibility**: Models trained with mixed precision (float16) must be run with the same TensorFlow version used for training. Iteration 2 model was trained with TF 2.18.0.

## Architecture Details

### U-Net with SE Blocks

- **Encoder**: 6 downsampling blocks (64 → 128 → 256 → 512 → 512 → 512 filters)
- **Decoder**: 5 upsampling blocks with skip connections (mirrors encoder)
- **SE Blocks**: Applied on deep layers (x4, x5, x6, x5b, x4b) for channel attention
- **Normalization**: InstanceNormalization (better than BatchNorm for small batches)
- **Residual Connections**: Within encoder/decoder blocks for gradient flow

### Data Pipeline

**DataLoader** (`provided_code/data_loader.py`):
- Modes: `training_model` (CT+dose), `dose_prediction` (CT only), `evaluation` (dose+masks)
- Caching: Pre-stacks all patients into RAM for instant batch slicing
- Normalization: CT [0,4095] → [0,1], Dose / 70 Gy

**Key methods**:
- `set_mode(mode)`: Configure required data fields and trigger caching
- `get_batches()`: Iterator yielding `DataBatch` objects
- `get_patients(patient_list)`: Load specific patients by ID
