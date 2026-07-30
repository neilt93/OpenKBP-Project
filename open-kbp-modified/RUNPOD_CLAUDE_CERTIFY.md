# RunPod Claude Brief — Certified robustness for photon dose prediction

You are a Claude Code instance running ON a RunPod 4090 box (network volume at `/workspace`).
This session turns the *empirical* noise defence into a *provable* one. Two jobs, in order:

1. **Bridge experiment (cheap, do first):** run the ADAPTIVE (EOT) attack on the existing
   test-time noise defence and measure how much of the 60-104% recovery survives an attacker
   that knows the defence is there. This is the honest motivation slide for the whole story.
2. **Certification:** median randomised-smoothing certificate on the best model — per-voxel
   certified dose intervals (Gy) and certified DVH intervals under an L2 CT-perturbation ball.

All code is built and the pure certificate math is unit-tested off-GPU. You are executing
runs, watching the gotchas below, and reporting numbers.

---

## 0. Environment & one-time bootstrap (fresh pod, NO network volume this session)
- Any box with 1x RTX 4090 24 GB + TF 2.18.0 (must match training — mixed-precision models are
  version-sensitive). Batch 1 for both scripts; do NOT raise it. ~30 min total, inference only.
- No network volume today (no 4090 stock in the volume's region), so work on the pod's
  container disk. Nothing persists after terminate — **copy results off before you stop** (§3).

Bootstrap the pod (copy-paste; fill in the three secrets):
```bash
# 1. Claude Code (native installer; alt: npm i -g @anthropic-ai/claude-code)
curl -fsSL https://claude.ai/install.sh | bash
export ANTHROPIC_API_KEY=...          # headless auth — no browser on the pod

# 2. Our code
cd /workspace && git clone https://github.com/neilt93/OpenKBP-Project.git openkbp
cd openkbp/open-kbp-modified && git checkout adversarial-retraining

# 3. Validation data — from the PUBLIC OpenKBP repo (our data is gitignored, not in our repo)
git clone https://github.com/ababier/open-kbp.git /tmp/okbp-data
DATA=/tmp/okbp-data/provided-data/validation-pats

# 4. Model — pull from the RunPod S3 volume bucket (reachable from any region)
#    NOTE: awscli is NOT preinstalled on the base pod — install it first (pip below, or apt).
pip install awscli
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
aws s3 cp s3://5jwj898h77/models/epoch_100.keras models/epoch_100.keras \
    --region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io

# 5. Python deps + sanity check
pip install "tensorflow[and-cuda]==2.18.0" pandas numpy scipy tqdm more_itertools
python tests/test_smoothing_certify.py       # 7 certificate-math tests must pass
```

The uploaded model is the **baseline** (`epoch_100.keras`, DVH 2.536) — the only one saved to
S3. The stronger DVH0.02 single model (2.049) was lost with the old volume; retrain it with
`runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --batch-size 4 --ptv-weight 4.0
--dvh-weight 0.02` if you want to certify the better model. Pass `--data-dir $DATA` to both
scripts below so they find the validation set (it is NOT at the default in-repo path).

---

## 1. Bridge experiment — adaptive attack on the noise defence  (~10-15 min for 10 patients)

```bash
MODEL=models/epoch_100.keras          # baseline pulled from S3 in §0 (step 4); $DATA from step 3
python adversarial_adaptive.py --model $MODEL --data-dir $DATA \
    --epsilons 0.02,0.05 --pgd-steps 10 --eot-samples 8 \
    --defense-sigma 0.1 --defense-samples 8 \
    --n-patients 10 --output adaptive_results/
```

Reads out four conditions per epsilon: clean, non-adaptive PGD (defended), and **adaptive
EOT-PGD (defended)**. The headline is the last two columns of the summary table:
`nonadapt rec%` vs `ADAPTIVE rec%`.

**What to expect / how to read it.** If adaptive recovery collapses far below non-adaptive
(e.g. 80% -> 10-20%), the empirical defence is largely obfuscated gradients (Athalye 2018) —
the expected, publishable result that motivates certification. If adaptive recovery stays
high, the noise defence is stronger than assumed; either way it is a real finding. `--defense-sigma
0.1` matches the best sigma from `SESSION_RESULTS.md`; if you have time, also try `0.05`.

**Cost knob:** EOT is `pgd-steps x eot-samples` forward+backward passes per patient per eps
(here 10x8 = 80). Drop `--eot-samples` to 4 for a faster first look; raise to 16 for a tighter
gradient estimate if the adaptive attack looks too weak (an under-powered EOT gradient can
*flatter* the defence — rule that out before concluding the defence holds).

---

## 2. Certification — median randomised smoothing  (~10-20 min for 10 patients, n=100)

```bash
python certify_smoothing.py --model $MODEL --data-dir $DATA \
    --sigma 0.05 --n-samples 100 --batch-draws 8 \
    --radii 0.5,1.0,2.0 --tol-gy 1.0 --alpha 0.001 \
    --n-patients 10 --output certify_results/
```

Produces, per L2 radius R: per-voxel certified dose-interval widths (median / p95 / max, Gy),
the fraction of in-body voxels certified within `--tol-gy`, and certified DVH-metric intervals.
`certify_summary.json` has per-patient detail incl. `dvh_interval_widths_gy`.

**How to read the certificate.** Narrower interval = stronger guarantee. `~HU/vox` in the
table is the per-voxel RMS of the L2 ball (R/sqrt(V) x 4095) — the honest per-voxel size of
the perturbation being certified against. Expect the whole-volume L2 ball to look small per
voxel; that tension is inherent to smoothing and should be reported, not hidden.

**Tuning sigma (the key trade-off).** Larger `--sigma` certifies a larger radius but widens
the intervals (less accurate under noise). Sweep `--sigma 0.02,0.05,0.10` (separate runs,
separate `--output` dirs) to map the accuracy/robustness curve. `--n-samples` controls
confidence tightness and cost linearly; 100 is a fast first pass, use 500-1000 for a final
number. `--alpha 0.001` = 99.9% confidence.

**Uncertified flag:** if a run prints `certified_low/high = False` for some radius, the sample
count is too small for that radius/sigma (the order-statistic rank fell off the end) — raise
`--n-samples` or lower the radius. The code clamps to the data range and flags it rather than
emitting a fake bound.

---

## 3. Report back + SAVE RESULTS (no network volume today = pod disk is ephemeral)
For each script: the summary table, the JSON path, and one sentence on the finding. For the
bridge experiment specifically, state the non-adaptive vs adaptive recovery gap — that single
number decides whether the certified-defence framing is motivated.

**Before terminating the pod, push the JSON results to S3 (they are small and will otherwise be
lost with the container disk):**
```bash
aws s3 cp adaptive_results/ s3://5jwj898h77/results/adaptive_results/ --recursive \
    --region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io
aws s3 cp certify_results/ s3://5jwj898h77/results/certify_results/ --recursive \
    --region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io
```
(The bucket persists even though no pod/volume is mounted — that is why the S3 upload is the
durable path this session. Fetch them back locally later with the reverse `aws s3 cp`.)

## 4. Phase 2 (only if 1-2 land and there is GPU budget)
Fine-tune the base model on Gaussian-noised CTs (SmoothAdv, Salman et al.) so it predicts well
UNDER the smoothing noise, then re-certify — this is the lever that widens the certified radius.
That is a `runpod_train.py` change (add train-time input noise at `--sigma`), not built yet;
flag it back rather than improvising.
