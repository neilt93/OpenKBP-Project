# RunPod Claude Brief — Test-time defence + Gao replication

You are a Claude Code instance running ON a RunPod 4090 box (EU-RO-1, network volume at
`/workspace`, container disk 30 GB). Two jobs this session, do them in order:

1. **Test-time augmentation defence** (already built, just run it): does augmentation applied
   at INFERENCE, with no retraining, blunt an adversarial attack on the CT input?
2. **Replicate Gao et al. 2025** accuracy: train with the (already implemented) DVH-aware loss
   and a 5-seed ensemble to push clean DVH score from 2.535 toward the ~1.4-1.5 tier.

All code and design are done. You are executing runs, watching for the gotchas below, and
reporting numbers back.

---

## 0. Hardware / environment
- GPU: 1x RTX 4090 24 GB. Batch 4 at 128^3, mixed precision. Do NOT raise batch size (OOM).
- RAM ~46 GB, 8 vCPU. No injection this session, so the data cache is only ~240 patients
  (~10 GB) and RAM is not tight.
- Everything durable lives on the network volume `/workspace` (survives pod terminate).

```bash
cd /workspace
git clone <repo-url> openkbp && cd openkbp
git checkout adversarial-retraining          # branch with all this work
pip install tensorflow[and-cuda]==2.18.0 pandas numpy scipy tqdm more_itertools pyyaml
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  # must list the 4090
```
**TF MUST be 2.18.0** — the float16 checkpoints are version-sensitive.

## 1. Get the two private models (the user sends them from their Mac)
The OpenKBP data is public and you pull it on-box (step 2). The trained models are private and
come from the user's SanDisk via `runpodctl`. The user runs `runpodctl send <file>` on their
Mac and gives you the one-time code; you receive:
```bash
mkdir -p /workspace/openkbp/open-kbp-modified/models_in
cd /workspace/openkbp/open-kbp-modified/models_in
runpodctl receive <code-for-epoch_100>   # baseline, DVH 2.535 / Dose 3.731 (~1.7 GB)
runpodctl receive <code-for-epoch_125>   # robust fine-tune, DVH 2.059 / Dose 3.595 (~343 MB)
```
Keep both. `epoch_100` is the baseline for BOTH tasks; `epoch_125` is the robust model for the
defence comparison only.

## 2. Pull the public OpenKBP data on-box
Sparse-checkout train + validation only (skip test-pats to save disk) into
`open-kbp-modified/provided-data/`. Verify the split: train = pt_1..200, validation = pt_201..240.
The defence run needs `validation-pats`; the replication training needs `train-pats`.

---

## TASK A — Test-time defence (do this first, it is quick)

Orchestrator is `adversarial_defense.py` (committed 5a2d94d). It attacks the bare model
(FGSM/PGD), applies each test-time transform (smooth / noise / intensity / flip), and scores
with the real OpenKBP DoseEvaluator in memory. Design notes are in the file's docstring and in
`ADVERSARIAL_RETRAINING.md`.

**Smoke test on ONE patient with the real model first** (catches load/shape issues before scaling):
```bash
cd /workspace/openkbp/open-kbp-modified
python adversarial_defense.py \
    --model models_in/epoch_100.keras \
    --attack fgsm --epsilons 0.02 \
    --defenses none smooth --n-patients 1 --quick \
    --output adversarial_defense_results/smoke/
```
If that prints a table with sane numbers (clean DVH near 2.5, attacked DVH higher), scale up.

**Baseline model, full first pass** (run epoch_100 BEFORE epoch_125 — the `smooth` defence is
essentially the P4 blur epoch_125 was trained to absorb, so on the robust model a "free" smooth
result is partly trained-in, not a pure test-time effect; interpret the two models separately):
```bash
python adversarial_defense.py \
    --model models_in/epoch_100.keras \
    --attack fgsm pgd --epsilons 0.02,0.05 \
    --defenses none smooth noise intensity flip \
    --n-patients 10 --output adversarial_defense_results/epoch100/
```
Then the same for the robust model:
```bash
python adversarial_defense.py \
    --model models_in/epoch_125.keras \
    --attack fgsm pgd --epsilons 0.02,0.05 \
    --defenses none smooth noise intensity flip \
    --n-patients 10 --output adversarial_defense_results/epoch125/
```
Read `defense_summary.json` in each output dir. The headline column is `recovered_frac_dvh`
(fraction of attack damage a defence recovers) alongside `clean_cost_dvh` (accuracy the defence
sacrifices on clean input). A defence only "works" if it recovers loss WITHOUT wrecking clean
accuracy. This is a NON-adaptive attacker (defence-unaware) — do not overclaim; adaptive/BPDA
is the rigorous follow-up.

Optional off-GPU sanity before running: `python tests/test_defense_transforms.py` and
`python tests/test_defense_scoring.py` (11 tests, need `more_itertools tqdm` installed).

---

## TASK B — Replicate Gao accuracy (DVH loss + ensemble)

The DVH-aware loss is ALREADY implemented (`--use-dvh`, histogram-percentile D_99/D_95/D_1 on
PTVs, wired through `network_functions.py`). It has never been run. Goal: reproduce the paper's
tier by training the best config WITH DVH loss, then ensembling.

Order matters — measure the single-model DVH-loss gain first, THEN spend on the ensemble.

**B1. One model, best config + DVH loss** (from scratch, 100 epochs). `--use-dvh` forces
`--no-jit` internally (the loss uses `tf.boolean_mask`, which breaks XLA):
```bash
cd /workspace/openkbp/open-kbp-modified
python runpod_train.py \
    --filters 64 --epochs 100 --use-se --use-aug \
    --use-dvh --dvh-weight 0.1 \
    --batch-size 4 --ptv-weight 4.0 --no-jit
```
The run auto-predicts on validation and scores at the end (results.json). Compare its DVH/Dose
to the baseline **2.535 / 3.731**. If DVH improves, DVH loss is pulling its weight; if it makes
things worse or unstable, drop `--dvh-weight` to 0.05 and rerun. Time epoch 1 first to estimate
wall-clock (DVH loss + no-jit is slower per epoch).

**B2. 5-seed ensemble** (only after B1 looks good — this is ~5x the cost). Repeat the best
config (with or without DVH loss, whichever B1 showed is better) across seeds, then average the
validation predictions:
```bash
for seed in 1 2 3 4 5; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug \
      --use-dvh --dvh-weight 0.1 --batch-size 4 --ptv-weight 4.0 --no-jit --seed $seed
done
```
Average the per-voxel predicted dose across the 5 models on validation, then score once. The
original competition winner used an ensemble; this is the single biggest lever we have not yet
pulled. Target: DVH toward ~1.5, Dose toward ~2.4.

---

## Guardrails (do not violate)
- **NEVER train on the validation set (pt_201..240).** Only train-pats (pt_1..200). No injection
  this session, so there is nothing to inject — ignore the `--inject-*` flags entirely.
- **TF must be 2.18.0.** Keep **batch size 4**. Use **`--no-jit`** (required for aug and forced
  by DVH loss anyway).
- If augmentation starves the GPU (low util in `nvidia-smi`, CPU pinned), the prefetch path from
  commit 5bd8ecb should already handle it; time epoch 1 before committing to all 100.
- Save everything to `/workspace` (the persistent volume). Do NOT rely on container disk.

## Report back to the user
- **Task A:** for each model (epoch_100, epoch_125), which defence recovered the most attack
  damage (`recovered_frac_dvh`) and at what clean cost. One sentence on whether test-time aug
  meaningfully blunts the attack.
- **Task B:** single-model DVH/Dose with DVH loss vs baseline 2.535/3.731; ensemble DVH/Dose;
  how close to the Gao ~1.4-1.5 tier.
- Wall-clock + cost for each. Final model paths on `/workspace` and the exact commands used.
