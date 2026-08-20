# Experiment plan — certified robustness for photon dose prediction

Dependency-ordered. Everything gates forward; nothing has a clock on it. Two load-bearing
disciplines throughout: **one difference per claim**, and **rules written before results exist**
(see `DECISION_RULES.md`). Status legend: ✅ done · ⏳ pod/GPU-gated · 🔬 off-GPU analysis.

## Phase 0 — Fix the instrument
Nothing downstream is trustworthy until these land.

1. ✅ **Decouple the flag from the code path.** `--aug-path {auto,tf,numpy}` in `runpod_train.py`
   makes the augmentation pipeline explicit; `--aug-noise` can no longer silently switch it, and
   `--aug-noise 0` means literally zero on the chosen path. `--aug-path numpy --aug-noise 0` is a
   clean zero-noise matched control (retires the `1e-9` hack). A one-time `=== AUG PATH: ... ===`
   banner is printed in `network_functions.train_model`, and the tf path is now labelled `AUGTF`
   in the run name (numpy stays `AUGGEO`) so a control-vs-arm mismatch is visible from the dir name.
2. ⏳ **Restore the baseline as a pinned artifact.** Check S3 for `epoch_100.keras`; if present,
   commit its sha256; if gone, retrain deterministically from committed code (fixed seed, logged
   config), push to S3, commit the sha256. "The baseline" must be a pinned artifact, not a memory.
3. ✅ **Cheap repairs.** `compare_certs.py`: `--self-test` restored (validates the reduction on
   committed certs, no GPU), docstring corrected (DVH widths recomputed vs frac read from stored
   aggregate; baseline is the confounded tf-path model — compare against the matched control).
   `smoothing_certify.certify_from_samples`: `np.partition` instead of `np.sort` (O(n) vs
   O(n log n) per voxel, bit-identical values). Radius guard: `max_certifiable_radius(n,σ,α)` =
   `σ·Φ⁻¹((α/2)^(1/n))` (1.45σ at n=100, 2.17σ at n=500); `certify_smoothing.py` drops radii above
   it by default (`--allow-vacuous-radii` to override), and the driver grids no longer append the
   vacuous `0.5/1/2` tail.
4. ✅ **Pre-commitment file** `DECISION_RULES.md`: what counts as a real effect (magnitude gate:
   2× paired-seed spread; accuracy gate: clean DVH/Dose within tolerance of Arm A), written before
   any run.

## Phase 1 — Tier 0: the accuracy gate ⏳ (inference only, no training)
Score plain DVH/Dose for every existing model: the three SmoothAdv arms, the matched control, the
seed replicate. One forward pass each (`runpod_train.py --eval-only`). This decides whether the
surviving σ=0.05 D95 signal and the ~25% aug-path tightening came from models that stayed accurate
or went flat. If the σ=0.05 models went flat → drop Arm B priority. If the control went flat → the
pipeline-effect candidate dies here, cheaply. **No other claim survives without these numbers.**

## Phase 2 — The seed grid ⏳
Four arms through the identical fixed pipeline, paired seeds 1–3 (same seed compared across arms so
seed luck cancels). Arms A/B/C/D defined in `DECISION_RULES.md`:
- **A** numpy path, zero noise — reference.
- **B** numpy path, σ=0.05 (σ=0.02 already known harmful; skip).
- **C** frozen restored baseline (no training) — the "did continued training itself do it" anchor.
- **D** small-ε adversarial training: PGD 2–3 steps at ε=0.02, random-init — also Birjoo's June ask.

Every run **exfils four things before its pod can die**: model weights, training log, clean
DVH/Dose score, certification JSONs (σ=0.05, informative radius set). Nothing lives only in a log.
Budget: D costs ~3–4× per step (treat as three runs). Escape hatch in `DECISION_RULES.md`.

## Phase 3 — Analysis 🔬 (off-GPU, comparisons matched to claims)
- **B vs A** — the noise-training effect (the only valid test of it).
- **A vs C** — the pipeline effect (the ~25% candidate).
- **D vs A** — the adversarial-training effect, as a curve over test ε (robustness decays past
  training ε; report the shape).
- **D vs test-time noise defense vs D+defense** — training-time hardening vs inference-time
  smoothing vs both, same 40 patients. The lever comparison nobody has run on dose prediction.
- Wilcoxon across patients on seed-averaged values; mean and worst-case-patient widths; every
  certificate number printed next to its clean-accuracy number. Verdicts from the pre-committed
  rules, not the vibe of the table.

## Phase 4 — The collective-DVH certificate 🔬 (math, no GPU)
Re-derive the DVH push-through without the all-voxels-simultaneously assumption, using the
percentile-slack structure of D95 (a percentile only moves when many voxels move together).
Re-analyze the **existing** certification samples under the tighter bound. If it works, every
certified interval in the paper tightens for free and limitation #4 (loose push-through) becomes a
contribution.

## Phase 5 — Write
Skeleton exists in this directory. Assembly order: certified DVH intervals (established) → the
confound anatomy as a methods lesson (drafted; keep it) → the grid's verdict on which lever pays
(any outcome is a result) → the empirical-vs-certified gap (~1000×: empirical defense resists L2
≈29 while the certified L2 radius is ≈0.028) as the honest frame → limitations scoped to L2, with
deformation certification named as future work.

## Parked, on purpose
- SmoothAdv variants — controls already answered it.
- Denoised smoothing — one optional later shot at the radius, only if a larger radius is still
  wanted.
- Deformation / contour certification — **paper 2** (the clinically-native threat model); do not
  let it leak into this one.
- Gamma-index and contour figures — unblocked by Phase 0's baseline restore; slot into Phase 5 as
  figures, not experiments.

## Key facts carried forward
- Aug-path confound + large seed floor fully explain the brainstem −27%; nothing left for
  noise-training. σ=0.05 D95|PTV70 ~−10% is the only survivor, gated on Phases 1–2.
- σ=0.02 noise-training is actively harmful (survives control + floor).
- Empirical (EOT-PGD) defense holds — recovers ~85% at ε=0.02, ~74% at ε=0.05 — but only the tiny
  certified radius is *provable*; foreground that gap.
- Operational gotchas (quota, resumable drivers, batch-draws 16, absolute `/workspace/results`)
  live in the Claude memory `runpod-workflow-gotchas`.
