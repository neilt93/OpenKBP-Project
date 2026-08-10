# RunPod Claude Brief — SmoothAdv (noise-trained certification), GIT-ONLY (no persistent storage)

You are a Claude Code instance in the terminal of a RunPod 4090 pod. **There is NO persistent
storage / S3 this session** — the disk is ephemeral and there is no bucket to push to. So:
exfil results via the FINAL REPORT (paste the numbers) and, if a GitHub token is available, by
committing the small JSONs to the repo. Do NOT rely on `aws s3` — the old bucket is gone.

Goal: the pivotal SmoothAdv experiment — train ON the smoothing noise, then re-certify, and
compare to the baseline certificates already in the repo. ~3 h. This needs ONLY code (git) +
public training data; it does NOT need the old baseline model or any S3.

## 0. Setup
```bash
cd /workspace && git clone https://github.com/neilt93/OpenKBP-Project.git openkbp 2>/dev/null
cd openkbp/open-kbp-modified && git checkout adversarial-retraining && git pull
pip install "tensorflow[and-cuda]==2.18.0" pandas numpy scipy tqdm more_itertools
git clone https://github.com/ababier/open-kbp.git /tmp/okbp-data
export DATA=/tmp/okbp-data/provided-data/validation-pats
export TRAINDATA=/tmp/okbp-data/provided-data
python tests/test_smoothing_certify.py && python tests/test_gamma_index.py   # both must pass
nvidia-smi   # confirm the 4090
```

## 1. SmoothAdv training — 3 noise-augmented models (~60–75 min)
`--aug-noise` injects Gaussian CT noise during training (train with the noise you smooth with).
The aug path needs `--no-jit`. `--out-suffix` keeps the three σ models in distinct dirs.
```bash
for S in 0.02 0.05 0.10; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 \
      --ptv-weight 4.0 --no-jit --aug-noise $S --data-dir $TRAINDATA --out-suffix smoothadv_s$S 2>&1 | tail -5
done
# models land in results/64filter_100epoch_SE_AUG_AUGGEO_MASK_PTV4.0_NORM_smoothadv_s<σ>/models/epoch_100.keras
```
(The trained model WEIGHTS are large and cannot be saved without storage — that is fine, we only
need the certificate JSONs below. Do not try to keep the .keras files.)

## 2. Re-certify each noise-trained model at its matching σ (~2 h) — THE HEADLINE
```bash
for S in 0.02 0.05 0.10; do
  M=$(ls -d results/*smoothadv_s$S*/models/epoch_100.keras | head -1)
  python certify_smoothing.py --model $M --data-dir $DATA --sigma $S --n-samples 100 \
      --batch-draws 8 --tol-gy 1.0 --alpha 0.001 --output certify_smoothadv_s$S/
done
```
**Compare directly to the baseline certificates ALREADY IN THE REPO** (non-noise-trained):
`reports/certified_robustness/experiment_results/certify_s{0.02,0.05,0.10}_full/certify_summary.json`.
The question that makes the paper: at the same σ, do the noise-trained models give **tighter
certified DVH-interval widths** (esp. D95 PTV70, mean brainstem) and/or a **larger frac-voxels
≤1 Gy** than the baseline? Tighter = "here's the fix" (method paper); no change = honest limitation.

## 3. Exfil (no S3 — do BOTH)
- **Always:** in your final report, paste a table of the new SmoothAdv certificate numbers vs the
  baseline (per σ: mean D95 PTV70 width, mean brainstem width, frac≤1Gy). This is the deliverable.
- **If a GitHub token is configured** (`git push` works): copy the JSONs into the repo and push:
  ```bash
  mkdir -p reports/certified_robustness/experiment_results
  cp -r certify_smoothadv_s*/ reports/certified_robustness/experiment_results/
  git add reports/certified_robustness/experiment_results/certify_smoothadv_s*/
  git commit -m "SmoothAdv certification results (noise-trained models)" && git push
  ```
  If `git push` fails (no token), skip it — the pasted numbers are the fallback.

## Deferred this session (need the baseline model, which is gone with the storage)
Gamma-index, contour figures, and adaptive-hardening all load the old baseline `epoch_100.keras`,
which is no longer retrievable (S3 bucket deleted; model too big for git). SKIP them now; run them
in a future session once the baseline model is re-provisioned onto a pod.

## Gotchas
- No S3 this session — do NOT run `aws s3` (bucket 5jwj898h77 is deleted).
- `--aug-noise` requires `--no-jit`; ~20 min/model on the 4090 (aug path, no XLA) — expected, not a hang.
- TF must be exactly 2.18.0. Verify each long job is actually producing output before assuming it runs.
- Do NOT terminate the pod; report numbers first.
