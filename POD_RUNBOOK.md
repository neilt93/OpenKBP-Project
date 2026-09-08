# Pod runbook — conformal coverage numbers + ASTRO P4 GIF

Two pending pod tasks, both light (inference-only, no training). Copy-paste in order. Paths assume
the repo at `/workspace/openkbp`; adjust the root if different. All script paths resolve relative to
`open-kbp-modified/` (that's `project_root` in the eval/inference scripts).

## 0. Setup
```bash
cd /workspace/openkbp && git checkout adversarial-retraining && git pull
pip install tensorflow[and-cuda]==2.18.0 pandas numpy scipy tqdm more_itertools matplotlib pyyaml pillow
```
Needed on-box (NOT in git — pull from SanDisk / RunPod S3 / regenerate):
- **Model:** `open-kbp-modified/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras`
- **Ground truth:** `open-kbp-modified/provided-data/validation-pats/` (pt_201–240)
- **Predictions (for Task 1 Path A):** `open-kbp-modified/openkbp_hn_robustness/predictions/{baseline,P*/L*}/`

Set once:
```bash
MODEL=open-kbp-modified/results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras
```

## 1. Conformal coverage numbers
The enabler patch is already committed — `evaluate_metrics.py` now writes `per_patient_dvh`
(per-patient, per-criterion residual) into every metric JSON. You just need to regenerate the
metrics so that field exists, then run the conformal engine.

**Path A — predictions still on disk (fast, no GPU):**
```bash
cd open-kbp-modified/openkbp_hn_robustness
python evaluate_metrics.py                       # re-scores existing predictions -> per_patient_dvh
```
**Path B — predictions gone (regenerate; needs the model + val CTs):**
```bash
cd open-kbp-modified/openkbp_hn_robustness
python generate_perturbed_data.py                # rebuild perturbed CTs (data_perturbed/)
python run_inference.py --model ../${MODEL#open-kbp-modified/}   # predict all 26 conditions
python evaluate_metrics.py
```
Then the conformal analysis (pure CPU, seconds):
```bash
cd /workspace/openkbp/reports/conformal_dvh
python conformal_dvh.py --self-test              # sanity: coverage math (clean ~0.90, shift collapses)
python conformal_dvh.py --alpha 0.1 --cal-frac 0.5   # -> coverage_vs_severity.csv
python conformal_plots.py                        # -> fig_coverage_vs_severity.png + coverage_threshold_table.{md,csv}
```
Expected: baseline marginal ≈ 0.90 (validates exchangeability); P4 coverage collapses first;
intensity families hold near nominal.

## 2. ASTRO Panel 6 — P4 severity GIF
```bash
cd /workspace/openkbp
python reports/report2_ct_perturbation/astro_poster/generate_p4_gif.py \
    --model $MODEL \
    --data-dir open-kbp-modified/provided-data/validation-pats \
    --patient pt_205 --fps 2
# -> reports/report2_ct_perturbation/astro_poster/p4_gif/p4_sweep_pt_205.gif (+ frame PNGs)
```
Try 2–3 patients and pick the clearest larynx slice. (The script auto-slices at the larynx centroid.)

## 3. Verify before exfil
```bash
python -c "import json; d=json.load(open('open-kbp-modified/openkbp_hn_robustness/metrics/per_patient/baseline.json')); print('per_patient_dvh present:', 'per_patient_dvh' in d)"
ls -la reports/conformal_dvh/coverage_vs_severity.csv reports/conformal_dvh/fig_coverage_vs_severity.png
ls -la reports/report2_ct_perturbation/astro_poster/p4_gif/
```

## 4. Exfil (NO push credentials on the pod — bundle out, push from the Mac)
```bash
cd /workspace/openkbp
tar czf /workspace/pod_out.tgz \
    open-kbp-modified/openkbp_hn_robustness/metrics \
    reports/conformal_dvh/coverage_vs_severity.csv \
    reports/conformal_dvh/fig_coverage_vs_severity.png \
    reports/conformal_dvh/coverage_threshold_table.md \
    reports/conformal_dvh/coverage_threshold_table.csv \
    reports/report2_ct_perturbation/astro_poster/p4_gif
runpodctl send /workspace/pod_out.tgz          # note the code
```
On the Mac:
```bash
cd "/Users/neiltripathi/Documents/OpenKBP Project"
runpodctl receive <code>                        # writes pod_out.tgz
tar xzf pod_out.tgz && rm pod_out.tgz
git add open-kbp-modified/openkbp_hn_robustness/metrics reports/conformal_dvh reports/report2_ct_perturbation/astro_poster/p4_gif
git commit -m "Pod run: conformal coverage numbers + ASTRO P4 GIF"
git push
```
All artifacts are small (metrics JSONs + a few PNGs/CSVs + one GIF) — safe to commit. Models
(*.keras) stay gitignored.

## Notes / gotchas (from prior pod sessions)
- `/workspace` has a ~20 GB hard quota; `df` is useless. These tasks write almost nothing, so quota
  isn't a risk here — but don't leave large checkpoints around.
- `evaluate_metrics.py` reads paths from `configs/default.yaml` (validation-pats, predictions,
  metrics). Use `--config configs/eval_robust_m.yaml` if scoring the robust model instead.
- If `run_inference.py` OOMs, it's batch-size 1 already; reduce concurrent work, not batch size.
```
