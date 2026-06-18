# Adversarial / Robustness Retraining (photons)

Harden the photon dose predictor against the CT perturbations the robustness study flagged
(P4 resolution and P2 bone-shift were the weak spots) by (1) **injecting the pre-generated
perturbed-CT sets** into the training group and (2) adding **geometric augmentation**
(translation, rotation, scaling, elastic) on top of the existing flips + intensity scaling.

All new code is numpy/scipy + small training hooks; unit-tested off-GPU:
`python tests/test_augmentation.py` and `python tests/test_inject.py`.

## Patient split — RESOLVED (2026-06-18)

Confirmed the original `data_perturbed/` is the **validation split (pt_201–243)** — its
generator docstring says "perturbed copies of *validation* patient CT volumes". Injecting
those would leak the test set. **Fix applied:** regenerated the same families on the
**training CTs (pt_1–200)** into a separate dir via
`openkbp_hn_robustness/configs/train.yaml`:

```bash
python openkbp_hn_robustness/generate_perturbed_data.py --config openkbp_hn_robustness/configs/train.yaml
#  -> writes <warehouse>/.../openkbp_hn_robustness/data_perturbed_train/<family>/<level>/pt_<=200/
```

Each generated dir is a COMPLETE training sample: a real perturbed `ct.csv` + symlinked
clean dose/masks from the original patient — so inject with `--inject-reuse-existing` (no
re-composition needed). Safety net still on: the injector derives held-out ids from
`validation-pats`, never injects them, and raises loudly on 0-match.

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
    --inject-perturbed openkbp_hn_robustness/data_perturbed_train \
    --inject-glob '*/*/{pid}/ct.csv' \
    --inject-families P2 P4 --inject-levels L3 L4 --inject-max-per-patient 4 \
    --aug-translate 0.08 --aug-rotate 10 --aug-scale 0.1 --aug-elastic 3 --aug-noise 0.02
```
- `data_perturbed_train` is the training-split set; the original `data_perturbed` is
  validation-only — do NOT inject it.
- Default compose builds uniquely-named patient dirs (`pt_3__P2_bone_shift_L4`); the
  DataLoader keys patients by `path.stem`, so unique names are required (raw `pt_3` dirs
  would collide and collapse variants). `--inject-reuse-existing` also produces unique
  names — use it only if a perturbed dir carries its own (non-symlinked) dose/masks.
- **Caching + RAM:** the loader pre-stacks ALL training patients in RAM. The cache dtype
  is now compact (masks/possible_dose_mask `uint8`, ct/dose `float32`) = **~40 MB/patient**
  (was 218 MB as float64), so 1000 patients (1:4 injection) ≈ **40 GB** — fits a 64 GB box
  with caching on. Still keep the count sane via families/levels/max-per-patient; for very
  large injections use `--no-cache` (slower, disk-bound). (The old `train_data.npz`
  precomputed cache is float64 — don't reuse it; let the loader preload fresh.)

- `--inject-glob` must match the real layout (see `--inspect`); `{pid}` marks the patient id.
- `--inject-reuse-existing` if a perturbed dir already contains dose + masks (else CT is
  composed with the clean patient's dose/masks).
- Geometric flags turn on the numpy augmentation path (includes flips+intensity); needs
  `--no-jit`. Validation predictions/scoring are unchanged (clean validation set).

## Throughput — measured (Mac, 14 cores)

Geometric augmentation runs `scipy.ndimage` ops over 13 channels (1 CT + 10 masks + dose +
possible_dose_mask) × 128³ per sample. `augment_batch` parallelizes samples across CPU
threads (the scipy ops release the GIL), measured on a 14-core Mac with the full
geometric+elastic+noise params at batch 4:

| | per batch | per sample | epoch aug @ 1:4 inject (1000 pts) |
|---|---|---|---|
| serial (old) | 2.07 s | 0.52 s | 8.6 min |
| **threaded (now)** | **0.63 s** | **0.16 s** | **2.6 min** |

So ~2.6 min/epoch of aug at the recommended injection size — tolerable, but on a fast GPU
it still won't fully overlap. For the real 100-epoch run, **prefetch** augmentation (run it
in a background thread / `tf.data` parallel `map` while the GPU computes the previous batch)
to hide it entirely; that's a training-loop change to make on the GPU box. Injection itself
has no per-step cost (it only adds patient dirs). `augment_batch` is deterministic given a
seeded rng (pass one from the trainer for reproducible ensembles).

## Files

| File | Role |
|---|---|
| `provided_code/augmentation.py` | numpy/scipy geometric + CT-noise aug (fused single resample; masks nearest, CT/dose linear, all fields stay registered) |
| `provided_code/inject_perturbed.py` | compose perturbed-CT + clean-dose training dirs; leakage guard, family/level filters, per-patient cap, `--inspect` |
| `provided_code/network_functions.py` | training loop calls `augment_batch_geometric` when `aug_params` set (else the prior tf flip/intensity path) |
| `runpod_train.py` | `--aug-*` and `--inject-*` flags; leakage-safe wiring |
| `tests/test_augmentation.py`, `tests/test_inject.py` | off-GPU unit tests (registration, binary masks, CT-only noise, leakage guard, 0-match raise) |
