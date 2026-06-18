# Adversarial / Robustness Retraining (photons)

Harden the photon dose predictor against the CT perturbations the robustness study flagged
(P4 resolution and P2 bone-shift were the weak spots) by (1) **injecting the pre-generated
perturbed-CT sets** into the training group and (2) adding **geometric augmentation**
(translation, rotation, scaling, elastic) on top of the existing flips + intensity scaling.

All new code is numpy/scipy + small training hooks; unit-tested off-GPU:
`python tests/test_augmentation.py` and `python tests/test_inject.py`.

## ⚠️ BLOCKER — verify the patient split before any training run

The perturbation sets were generated for the **robustness study, which evaluated on the
held-out validation patients (pt_201–240)**. Injecting those into training **leaks the test
set** and invalidates every DVH/Dose score. Guards are in place, but confirm the split first:

```bash
# with the warehouse mounted:
python -m provided_code.inject_perturbed --perturbed-root <data_perturbed> --inspect
#  -> are the patient ids <= pt_200 (training, OK) or pt_201-240 (validation, NOT OK)?
```

- If the ids are **validation (pt_201–240)**: the existing sets are the **wrong split** for
  injection — you must **regenerate the perturbations on the TRAINING CTs** (pt_1–200).
- The injector **never** composes a held-out id (it derives them from `validation-pats`) and
  **raises loudly if 0 patients match** — so you cannot silently train on nothing, and you
  cannot "fix" a 0-match by pointing `--original-root` at `validation-pats` (that would leak).

## Decisions to make (shape the trained model)

- **Families:** all P1–P5 (chosen). P1 noise / P3 bias / P5 dental the model already handles;
  P2 bone-shift + P4 resolution are the real targets — consider `--inject-families P2 P4`.
- **Levels & ratio:** all families × all levels ≈ 5×6×200 ≈ 6000 perturbed vs 200 clean would
  swamp the clean signal, and extreme levels (L5) can teach the model to *ignore* the CT.
  Use `--inject-levels` and `--inject-max-per-patient` to set a deliberate clean:perturbed
  ratio (e.g. ~1:2). **Pick this consciously.**

## Run (RunPod, GPU, warehouse mounted)

```bash
python runpod_train.py \
    --filters 64 --epochs 100 --use-se --batch-size 4 --ptv-weight 4.0 --no-jit \
    --inject-perturbed openkbp_hn_robustness/data_perturbed \
    --inject-glob '*/*/{pid}/ct.csv' \
    --inject-families P2 P4 --inject-levels L3 L4 --inject-max-per-patient 4 \
    --aug-translate 0.08 --aug-rotate 10 --aug-scale 0.1 --aug-elastic 3 --aug-noise 0.02
```

- `--inject-glob` must match the real layout (see `--inspect`); `{pid}` marks the patient id.
- `--inject-reuse-existing` if a perturbed dir already contains dose + masks (else CT is
  composed with the clean patient's dose/masks).
- Geometric flags turn on the numpy augmentation path (includes flips+intensity); needs
  `--no-jit`. Validation predictions/scoring are unchanged (clean validation set).

## ⚠️ Throughput — measure before a full run

Geometric augmentation runs `scipy.ndimage.map_coordinates` over 13 channels (1 CT + 10
masks + dose + possible_dose_mask) × 128³ per sample, in the training loop on CPU (~1–3 s/
sample). Over 100 epochs that can dwarf GPU time. **Time one epoch first.** If it starves the
GPU, move augmentation into a prefetched/parallel path (tf.data parallel `map`, or loader-side
multiprocessing) before committing to the full run. Injection has no such cost (it only adds
patient dirs).

## Files

| File | Role |
|---|---|
| `provided_code/augmentation.py` | numpy/scipy geometric + CT-noise aug (fused single resample; masks nearest, CT/dose linear, all fields stay registered) |
| `provided_code/inject_perturbed.py` | compose perturbed-CT + clean-dose training dirs; leakage guard, family/level filters, per-patient cap, `--inspect` |
| `provided_code/network_functions.py` | training loop calls `augment_batch_geometric` when `aug_params` set (else the prior tf flip/intensity path) |
| `runpod_train.py` | `--aug-*` and `--inject-*` flags; leakage-safe wiring |
| `tests/test_augmentation.py`, `tests/test_inject.py` | off-GPU unit tests (registration, binary masks, CT-only noise, leakage guard, 0-match raise) |
