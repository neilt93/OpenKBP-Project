# Multi-model robustness comparison plan

Goal: determine whether the headline finding (resolution loss is the only clinically visible
perturbation family, first crossing 1.0 Gy at L2 with larynx D0.1cc as sentinel) is specific to our
SE U-Net or a property of the dose-prediction task. This upgrades the paper from a single-model
case study to an architecture survey, and gives a robustness-vs-accuracy scatter across models.

## Why this is cheap

The perturbation battery and evaluation are already model-agnostic:
- The 26-condition perturbed test set (baseline + 5 families x 5 levels x 40 patients) is already
  generated and frozen on the SanDisk warehouse (`data_perturbed/`), deterministic seeds.
- Evaluation (`evaluate_metrics.py`) consumes prediction CSVs only; it does not care what produced
  them.
- Per model, the marginal work is 1040 inference passes (26 x 40) plus training if no weights
  exist. Inference for all 26 conditions took well under an hour on a 4090 for the SE U-Net.

## Candidate model set (pick 4-5)

| # | Model | Framework | Weights | Cost |
|---|---|---|---|---|
| 1 | SE U-Net (ours, done) | TF 2.18 | trained | none, results exist |
| 2 | Plain 3D U-Net (no SE, no residual, BatchNorm) | TF 2.18 | retrain properly, 100 epochs | ~1 GPU-day equiv on 4090 (see gpu-usage notes) |
| 3 | Cascade / coarse-to-fine U-Net (OpenKBP winner style, e.g. C3D) | PyTorch | open-source repo exists (LSSC/C3D); retrain or adapt released weights to our split | adaptation effort > compute |
| 4 | Swin UNETR or UNETR (transformer encoder) | PyTorch (MONAI) | train from scratch on the 200-patient split | 1-2 GPU-days; MONAI gives the architecture for free |
| 5 | HD U-Net or DenseNet-style dose predictor (literature standard) | TF or PyTorch | train from scratch | ~1 GPU-day |

Notes:
- 2 vs 1 isolates the effect of attention/residual tricks on robustness at matched data/loss.
- 4 tests the inductive-bias question: transformers have weaker locality bias; do they degrade
  differently under blur?
- 3 tests whether ensembling/cascading (higher accuracy) buys robustness or just accuracy.
- Do NOT reuse the old 11.5-DVH "baseline" run as model 2; it was undertrained and would confound
  robustness with accuracy.

## Protocol (identical for every model)

1. Train (if needed) on the standard 200-patient training split. Freeze recipe per model family:
   100 epochs, masked MAE loss, PTV weight 4.0 where applicable; record any deviation.
2. Predict the 40 test patients under all 26 conditions -> prediction CSVs in
   `predictions_<model>/<family>/<level>/pt_2xx.csv` (same layout as now).
3. Run `evaluate_metrics.py` per model -> `summary.csv` + per-criterion tables.
4. Extract per model: (a) baseline dose/DVH score; (b) worst-criterion curve per family;
   (c) first crossing level of the 1.0 Gy threshold; (d) sentinel criterion identity.

## Analyses

- Threshold table per model (the 4-green-1-red table): does P4-only sensitivity replicate?
- Crossing-point comparison: does L2 move? A model whose crossing is L3+ is operationally more
  deployable; one crossing at L1 is worse than ours.
- Sentinel stability: is larynx D0.1cc always the canary, or is the sentinel architecture-dependent?
- Robustness vs accuracy scatter: baseline DVH score (x) vs P4-L4 worst-criterion shift (y).
  Tests "accurate models are not necessarily robust models" quantitatively.
- Secondary: does any architecture become sensitive to an intensity family (e.g. transformers to
  bias field)?

## Execution order and gating

1. (pod, cheap) Model 2 plain U-Net: retrain + sweep. Same TF stack, zero porting work. This alone
   is a publishable ablation (SE/residual vs robustness).
2. (pod) Model 4 Swin UNETR via MONAI: port the data loader (CSV -> tensor is ~100 lines; cache
   as npz like train_data.npz), train, sweep.
3. (pod) Model 5 HD U-Net.
4. (stretch) Model 3 cascade; only if the released code adapts cleanly to the 128^3 CSV format.
5. Write up as the multi-architecture section of the paper (Section: Future work -> Results).

Pod rules per POD_RUNBOOK.md and the RunPod gotchas notes (abs /workspace/results paths,
fail-loud drivers, 20 GB /workspace quota: stream predictions to the network volume, do not stack
five models' checkpoints locally).

## Deliverable

One figure (5 threshold curves, one per model, P4 family) + one table (per-model crossing level,
sentinel, baseline score) + the robustness-vs-accuracy scatter. Slots into the paper as a new
Results subsection; the current single-model text already frames it in Future Work.
