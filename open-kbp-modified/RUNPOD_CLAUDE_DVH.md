# RunPod Claude Brief — Fixed DVH-loss re-run

You are a Claude Code instance on a RunPod 4090 (EU-RO-1), the SAME network volume as the
last session mounted at `/workspace`. One job: confirm the **fixed** DVH-aware loss now
improves the score instead of hurting it, and if so, whether it beats/complements the
no-DVH ensemble (current best DVH 1.911 / Dose 3.289).

Background: last session, `--use-dvh` at weight 0.1 REGRESSED the score (DVH 3.130 vs
baseline 2.536). Root cause was a mid-biased percentile estimator + scoring only sample 0
of each batch. Fixed in commit `d7f28a8` (exact `tf.sort` + linear-interp percentile,
averaged over the whole batch). Numpy mirror already proves the algorithm; this run confirms
it end to end on the GPU.

---

## 0. Setup (volume persists, so this is quick)
The repo, OpenKBP data, and models are already on `/workspace` from last session.
```bash
cd /workspace/openkbp
git pull                                   # must include commit d7f28a8 (the DVH fix)
pip install "tensorflow[and-cuda]==2.18.0" pandas numpy scipy tqdm more_itertools pyyaml
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  # 4090
```
If this is a FRESH volume (not the EU-RO-1 one) and `/workspace/openkbp` is missing: re-clone
(`git clone https://github.com/neilt93/OpenKBP-Project.git openkbp`), re-pull the public
OpenKBP data into `open-kbp-modified/provided-data/`, and have the user `runpodctl send` the
models again. Otherwise everything is already there.

## 1. Smoke-test the fix BEFORE training (seconds, no GPU cost)
TF is present now, so the TF half of the percentile test runs:
```bash
cd /workspace/openkbp/open-kbp-modified
python tests/test_dvh_percentile.py
```
Must print the TF PASS lines (differentiable_percentile matches np.percentile, gradient
flows, identical-dose error is zero). If it does not, STOP and report — do not train.

## 2. Stage 1 — single-model DVH sweep (measure the gain cheaply first)
Train the best config WITH the fixed DVH loss, one run per weight. Each is from scratch, 100
epochs, ~15-20 min on the 4090 (`--use-dvh` forces `--no-jit`; the flag also changes the
model dir name so runs do not collide):
```bash
for w in 0.02 0.05 0.1; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug \
      --use-dvh --dvh-weight $w --batch-size 4 --ptv-weight 4.0 --no-jit
done
```
Each run auto-predicts on validation and writes `results/<name>/results.json` with
dvh_score / dose_score. Compare to:
- **baseline (no DVH): DVH 2.536 / Dose 3.731**
- **no-DVH per-seed range: ~2.34-2.67** (a fair single-model comparison)

Verdict: the fix works if the best weight lands a single-model DVH **at or below ~2.4**
(clearly better than the old 3.130, ideally beating the no-DVH single models). Time epoch 1
first to confirm the ~15-20 min estimate.

## 3. Stage 2 — DVH-loss ensemble (ONLY if Stage 1 shows a gain)
If a weight beats the no-DVH single models, run that weight as a 5-seed ensemble and average
the validation predictions, same as last session's winning ensemble:
```bash
BEST_W=<the winning weight from Stage 1>
for s in 1 2 3 4 5; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug \
      --use-dvh --dvh-weight $BEST_W --batch-size 4 --ptv-weight 4.0 --no-jit --seed $s
done
python ensemble_predict.py \
    --models 64filter_100epoch_SE_DVH${BEST_W}_AUG_MASK_PTV4.0_NORM_seed{1,2,3,4,5} \
    --epoch 100 --output ensemble_5seed_dvh${BEST_W}
```
(Check the exact model dir names with `ls results/` — the DVH tag format is `DVH<weight>`.)
Compare the ensemble DVH/Dose to the current best **1.911 / 3.289**.

## Guardrails
- TF must be **2.18.0**. Batch **4**. **`--no-jit`** (forced by DVH loss anyway).
- **Never train on validation** (pt_201-240); training is pt_1-200 only. No injection this run.
- Save everything under `/workspace` (persistent). Push a short results file + `results.json`s.

## Report back
- Per-weight single-model DVH/Dose vs baseline 2.536/3.731 and vs the no-DVH seeds — did the
  fixed DVH loss help, and at which weight?
- If Stage 2 ran: DVH-loss ensemble vs the 1.911 no-DVH ensemble.
- Wall-clock + cost, final model paths, exact commands.
