# RunPod Claude Brief — SmoothAdv (noise-trained certification) + add-ons

You are a Claude Code instance on a RunPod 4090 (ephemeral container disk, no network volume).
Goal: run the pivotal SmoothAdv experiment (train ON the smoothing noise → re-certify, to widen
the certified radius) plus three add-ons. **Push every result dir to S3 the moment it finishes**
— the disk is ephemeral. Work the queue in order; don't stop on the first. ~4–5 h total.

## 0. Setup (same as RUNPOD_CLAUDE_CERTIFY.md §0)
```bash
cd /workspace && git clone https://github.com/neilt93/OpenKBP-Project.git openkbp 2>/dev/null
cd openkbp/open-kbp-modified && git checkout adversarial-retraining && git pull
pip install "tensorflow[and-cuda]==2.18.0" pandas numpy scipy tqdm more_itertools awscli
export ANTHROPIC_API_KEY=...  AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
git clone https://github.com/ababier/open-kbp.git /tmp/okbp-data && DATA=/tmp/okbp-data/provided-data/validation-pats
TRAINDATA=/tmp/okbp-data/provided-data            # train-pats + validation-pats
aws s3 cp s3://5jwj898h77/models/epoch_100.keras models/epoch_100.keras --region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io
python tests/test_smoothing_certify.py && python tests/test_gamma_index.py    # both must pass
S3="--region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io"
```

## 1. SmoothAdv training — the pivotal run (~60–75 min total)
Train three noise-augmented models, σ matching the certification sweep. `--aug-noise` injects
Gaussian CT noise during training (the Cohen-style "train with the noise you smooth with").
The aug path needs `--no-jit` (XLA off).
```bash
for S in 0.02 0.05 0.10; do
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 \
      --ptv-weight 4.0 --no-jit --aug-noise $S --data-dir $TRAINDATA \
      --out-suffix smoothadv_s$S 2>&1 | tail -5
done
# --out-suffix keeps the three σ models in distinct dirs (all other flags are identical;
# without it they would overwrite each other). Model dir: results/64filter_100epoch_SE_AUG_AUGGEO_MASK_PTV4.0_NORM_smoothadv_s<σ>/
```
Then push the trained models so they survive the pod:
```bash
for S in 0.02 0.05 0.10; do aws s3 cp results/ s3://5jwj898h77/models/smoothadv_s$S/ --recursive --exclude "*" --include "*smoothadv_s$S*epoch_100.keras" $S3; done
```

## 2. Re-certify each noise-trained model at its matching σ (~2 h)
The comparison that makes the paper: does noise-training TIGHTEN the certified DVH intervals /
widen the radius vs the non-noise-trained baseline (already in s3 certify_s*_full)?
```bash
for S in 0.02 0.05 0.10; do
  M=$(ls -d results/*smoothadv_s$S*/models/epoch_100.keras | head -1)
  python certify_smoothing.py --model $M --data-dir $DATA --sigma $S --n-samples 100 \
      --batch-draws 8 --tol-gy 1.0 --alpha 0.001 --output certify_smoothadv_s$S/
  aws s3 cp certify_smoothadv_s$S/ s3://5jwj898h77/results/certify_smoothadv_s$S/ --recursive $S3
done
```
REPORT: for each σ, mean certified DVH-interval width (esp. D95 PTV70, mean brainstem) and
frac-voxels≤1Gy, **side by side with the baseline** certify_s{0.02,0.05,0.10}_full. Tighter = win.

## 3. Gamma-index robustness (~30–45 min)
```bash
python compute_gamma.py --model models/epoch_100.keras --data-dir $DATA \
    --epsilons 0.02,0.05 --defense-sigma 0.1 --defense-samples 8 --n-patients 20 --output gamma_results/
aws s3 cp gamma_results/ s3://5jwj898h77/results/gamma_results/ $S3 --recursive
```
REPORT: mean GPR (undef + def) at 3%/3mm and 3%/2mm vs TG-218 (95% tol / 90% action).

## 4. Contour-safety figures (~15 min)
```bash
python save_contour_overlay_figures.py --model models/epoch_100.keras --data-dir $DATA \
    --patient-ids pt_201 pt_205 pt_210 --epsilons 0.02 --attack pgd --n-slices 3 --output contour_overlay_figures/
aws s3 cp contour_overlay_figures/ s3://5jwj898h77/results/contour_overlay_figures/ $S3 --recursive
```

## 5. Adaptive hardening (~30–45 min) — confirm the ε=0.05 leak isn't under-powered EOT
```bash
python adversarial_adaptive.py --model models/epoch_100.keras --data-dir $DATA \
    --epsilons 0.02,0.05 --pgd-steps 40 --eot-samples 32 --defense-sigma 0.1 --defense-samples 16 \
    --n-patients 10 --output adaptive_vstrong/
aws s3 cp adaptive_vstrong/ s3://5jwj898h77/results/adaptive_vstrong/ $S3 --recursive
```
REPORT: does recovery drop vs the earlier adaptive_strong (eot16/pgd20)? If stable → defense
genuinely holds; if it drops → the ε=0.05 leak grows with attack strength (state honestly).

## 6. Wrap up
Leave everything in s3://5jwj898h77/results/. Give a final table: SmoothAdv certified widths vs
baseline (per σ), gamma GPRs, adaptive-hardening recovery. Do NOT terminate the pod.

## Gotchas
- `--aug-noise` requires `--no-jit`; training is ~20 min/model on the 4090 (aug path, no XLA).
- `--out-suffix` (added for this run) tags the dir so the three σ models don't collide; the step-2
  glob `results/*smoothadv_s$S*` resolves each. If a dir is missing, read the exact name from the
  training log rather than guessing.
- TF must be exactly 2.18.0 (mixed-precision model). NaNs on load → wrong TF version.
- Ephemeral disk: if you didn't `aws s3 cp` a result, it does not exist. Push after each step.
