# RunPod Claude Brief — Photon Adversarial / Robustness Retraining

You are a Claude Code instance on a RunPod GPU box. Your job: **fine-tune the existing
photon dose-prediction U-Net to be robust to CT perturbations**, by injecting perturbed-CT
training samples + geometric augmentation, then measure that robustness improved without
hurting clean accuracy. All the code and design is done; you are executing the run.

Read `open-kbp-modified/ADVERSARIAL_RETRAINING.md` for the design rationale. This file is
the operational checklist.

---

## 0. Hardware (what the user provisioned)
- **GPU:** 1× 24 GB (RTX 4090 or 3090). Batch 4, 128³, mixed precision. Do NOT increase
  batch size (OOM). No A100/H100 needed.
- **RAM:** ≥ 64 GB (the data cache is ~40 MB/patient; ~1000 patients ≈ 40 GB).
- **CPU:** 8–16 cores help (augmentation is threaded).
- **Disk:** **≥ 40 GB recommended.** On a tight 32 GB box it still fits IF you generate
  ONLY the perturbations you inject (§2) and keep few checkpoints (`--keep-history 2`).
  The full P1–P5 × 5-level set (~17 GB) will overflow alongside TF/CUDA (~6 GB) + data +
  checkpoints — don't generate it.

Verify first: `nvidia-smi` (GPU visible), `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

## 1. Environment + code
```bash
cd /workspace && git clone <repo-url> openkbp && cd openkbp
git checkout adversarial-retraining            # <-- the branch with this work
pip install tensorflow[and-cuda]==2.18.0 pandas numpy scipy tqdm more_itertools pyyaml
```
TensorFlow MUST be 2.18.0 (the best model was trained with it; mixed-precision float16
checkpoints are version-sensitive).

## 2. Data (pull public data on-box; only the model comes from the Mac)
- **`provided-data/`** is the **public OpenKBP dataset** — pull it directly on this box
  (fast) into `open-kbp-modified/provided-data/`. Sparse-checkout `train-pats` +
  `validation-pats` only (skip `test-pats` to save disk). Source:
  `https://github.com/ababier/open-kbp` (verify split: train=pt_1–200, validation=pt_201+).
- **`epoch_100.keras`** (~1.7 GB) is private (the trained model) — the USER rsyncs it from
  their Mac to `open-kbp-modified/results/iteration2_ctmax4095/models/epoch_100.keras`.
  (Print this box's SSH host/port so they can run the rsync.)
- The perturbed sets on the user's Mac symlink to absolute paths and WON'T transfer — so
  regenerate them here.

Generate ONLY the perturbations you'll inject (disk!): P2 + P4 at L3,L4 (CPU, ~10 min):
```bash
cd open-kbp-modified
python openkbp_hn_robustness/generate_perturbed_data.py \
    --config openkbp_hn_robustness/configs/train_runpod.yaml \
    --perturbations P2_bone_shift P4_resolution --levels L3 L4
# ~800 CTs (~3 GB) = exactly what the recommended injection (§3) uses. Do NOT generate all
# P1-P5 x 5 levels (5000 CTs, ~17 GB) — it overflows a 32 GB disk and isn't injected.
```
Sanity-check it's the TRAINING split (must be pt_1..200, NOT pt_201+):
```bash
python -m provided_code.inject_perturbed --perturbed-root openkbp_hn_robustness/data_perturbed_train --inspect
```

## 3. Fine-tune (resume from the best model + inject + augment)
Recommended: **fine-tune ~25 epochs from epoch_100** (not 100 from scratch). `runpod_train`
auto-resumes from the highest existing checkpoint in the model dir, so point `--epochs` past
100 and it continues. Copy the baseline checkpoint into the NEW model's dir first so it
resumes from it:
```bash
# new model name is derived from flags; pre-seed it with the baseline weights to fine-tune
NEW=results/64filter_125epoch_SE_AUGGEO_INJ_MASK_PTV4.0_NORM/models
mkdir -p "$NEW" && cp results/iteration2_ctmax4095/models/epoch_100.keras "$NEW/"

python runpod_train.py \
    --filters 64 --epochs 125 --use-se --batch-size 4 --ptv-weight 4.0 --no-jit \
    --inject-perturbed openkbp_hn_robustness/data_perturbed_train \
    --inject-glob '*/*/{pid}/ct.csv' \
    --inject-families P2 P4 --inject-levels L3 L4 --inject-max-per-patient 4 \
    --aug-translate 0.08 --aug-rotate 10 --aug-scale 0.1 --aug-elastic 3 --aug-noise 0.02
```
- This injects ~800 perturbed patients (1:4 clean:perturbed). The leakage guard refuses any
  validation id and RAISES on 0-match — if it raises, STOP and re-check the split (§2).
- `--no-jit` is required (the numpy augmentation path runs on the CPU batch).
- Expect **~25 epochs**; injection makes each epoch ~5× the baseline (1000 vs 200 patients).
  Rough wall-clock: **~2–4 h** on a 4090. If `--epochs 125` is too slow, 115 (15 epochs) is
  a reasonable floor for fine-tuning.
- If you change the injection mix, the model NAME changes (AUGGEO/INJ tags + epoch count) —
  re-derive `$NEW` so the resume picks up the baseline.

**If augmentation starves the GPU** (watch `nvidia-smi` — low utilization, CPU pinned):
augmentation runs synchronously in the training loop (~0.16 s/sample threaded). Either
accept ~2.6 min/epoch overhead, or implement prefetch (see §6). Time epoch 1 before
committing to all 25.

## 4. Evaluate clean accuracy (must not regress much)
The run auto-predicts on validation + scores. Compare to baseline **DVH 2.535 / Dose 3.731**:
```bash
# results printed at end + saved to results/<name>/results.json
```
Robustness retraining usually costs a little clean accuracy — a small DVH/Dose increase is
acceptable; a large jump means the injection ratio/augmentation is too aggressive.

## 5. Measure the robustness GAIN (the actual goal)
Run the existing robustness eval (perturb VALIDATION CTs → predict → DVH degradation) for
BOTH the baseline and the new model, and compare. See
`openkbp_hn_robustness/RUNPOD_INSTRUCTIONS.md` for that pipeline. Success =
**the new model degrades LESS under P2 (bone-shift) and P4 (resolution)** than the baseline,
while clean DVH/Dose stay close. Report the degradation curves side by side.

## 6. (Optional) efficiency upgrades you can implement + test here
- **Prefetch augmentation** (hides the CPU aug cost): run augmentation for batch N+1 in a
  background thread while the GPU trains on batch N (a producer thread feeding a small
  queue, or a `tf.data.Dataset.from_generator(...).prefetch()`). Wrap the loop in
  `provided_code/network_functions.py:train_model`. Verify loss curve matches a short
  non-prefetched run.
- **Ensemble:** repeat §3 with `--seed 1..5` (augmentation is non-deterministic per run →
  natural diversity) and average predictions for the best scores (the baseline best used a
  5-seed ensemble). ~5× the time/cost.

## Guardrails / gotchas (do not violate)
- **NEVER inject the validation set.** The original `data_perturbed/` (if present) is the
  VALIDATION split (pt_201–243) — injecting it leaks the test set. Only inject
  `data_perturbed_train` (pt_1–200). The code guards this; don't override `--original-root`.
- **TF must be 2.18.0** (checkpoint compatibility).
- **Keep batch size 4** and **`--no-jit`**.
- Axis convention (for any orientation work): loader BDHWC is D=A-P, H=L-R, W=S-I; the new
  `provided_code/augmentation.py` is correct, the legacy `augment_batch_tf` axis labels are
  not (left unchanged on purpose).
- Run the test suites if you touch the code: `python tests/test_augmentation.py` and
  `python tests/test_inject.py`.

## Report back to the user
1. Clean DVH/Dose of the fine-tuned model vs baseline (2.535 / 3.731).
2. Robustness degradation under P2/P4: baseline vs fine-tuned (did it improve?).
3. Wall-clock + cost.
4. The final model path (download target) and the exact command used.
