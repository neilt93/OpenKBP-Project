# OpenKBP Dose Prediction + Adversarial Robustness

Deep-learning models for automated radiotherapy dose prediction on the
[OpenKBP Grand Challenge](https://github.com/ababier/open-kbp) head-and-neck
dataset (AAPM), plus an adversarial-robustness study of those models. The
robustness work is the basis of a SERA 2026 submission on how fragile clinical
dose predictors are to small, imaging-noise-scale perturbations.

## What it does

- Trains a 3D U-Net to predict a full 3D dose distribution from a patient CT
  and its ROI structure masks (128x128x128 volumes, 10 structures).
- Adds architectural and training improvements over the challenge baseline:
  Instance normalization, squeeze-and-excitation channel attention, residual
  connections, masked MAE loss, PTV voxel weighting, augmentation, mixed
  precision, and full-dataset caching.
- Evaluates adversarial robustness with FGSM and PGD attacks on the CT input
  across a sweep of perturbation magnitudes, and reports degradation.

## Results (this repo's best config)

| Model | DVH Score | Dose Score |
|-------|-----------|------------|
| This repo (SE + aug + PTV weighting) | ~2.5 | ~3.7 |
| Original baseline | 11.481 | 7.180 |
| Competition winner (ensemble + cascade) | 1.478 | 2.429 |

Lower is better. Exact numbers vary slightly by iteration and CT clip value;
see `open-kbp-modified/SESSION_RESULTS.md` and `reports/` for the logged runs.

## Run

Training is designed for a single GPU (RunPod, RTX 3090 class). TensorFlow
2.18.0 is pinned because the mixed-precision models must be loaded with the
version they were trained on.

```bash
# best training config
python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug \
  --batch-size 4 --ptv-weight 4.0 --no-jit

# adversarial evaluation
python adversarial_eval.py \
  --model results/<run>/models/epoch_100.keras \
  --attack fgsm pgd --epsilons 0,0.001,0.005,0.01,0.02,0.05,0.1 \
  --output adversarial_results/
python plot_adversarial.py --results-dir adversarial_results/
```

Perturbation scale: in normalized CT space `HU = epsilon * 4095`, so
epsilon=0.01 is about 41 HU, roughly the level of typical CT noise.

See `SETUP_GUIDE.md` and `open-kbp-modified/RUNPOD_SETUP.md` for full setup.

## Layout

| Path | Purpose |
|------|---------|
| `open-kbp-modified/` | Training, prediction, evaluation, adversarial scripts |
| `open-kbp-modified/provided_code/` | U-Net, data loader, DVH/dose scoring |
| `reports/` | Robustness and certified-robustness write-ups |
| `Copy_of_open_kbp_introduction.ipynb` | Challenge intro notebook |

## Limitations

- Not at ensemble/cascade competition-winner accuracy; this is a single-model
  pipeline.
- Trained models are TensorFlow-version-sensitive (float16 mixed precision).
- Adversarial attacks perturb the CT only, leaving structure masks fixed; they
  are a robustness diagnostic, not a clinical threat model.
- Large data and trained weights are not committed to the repo.
