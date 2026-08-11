# SmoothAdv certified-robustness run — findings (2026-08-10/11)

7 certificates, all 40 patients / n_samples=100 / alpha=0.001 / tol=1.0 Gy,
sigma-scaled radii matching the committed baselines (0.5x, 1.0x, 1.4x sigma).

| certificate | model | what it is |
|---|---|---|
| `certify_smoothadv_s{0.02,0.05,0.10}` | noise-trained, `--aug-noise {sigma}` | the three SmoothAdv arms |
| `certify_control_s{0.02,0.05,0.10}` | `--aug-noise 1e-9`, same AUGGEO path | matched control (kills the aug-path confound) |
| `certify_seed1_s0.05` | `--aug-noise 0.05 --seed 1` | seed replicate (measures the noise floor) |

## Headline: this is the "flat / limitation" outcome, not a method contribution

The apparent wins against the committed baseline do not survive controls.

### 1. The brainstem result is fully explained away

| stage | mean\|Brainstem width change |
|---|---|
| SmoothAdv vs committed baseline | **-27%** (looks like a big win) |
| ...of which the augmentation path alone (control vs baseline, ZERO noise) | **-18 to -25%** |
| remaining, SmoothAdv vs matched control | -9.4 / -10.0 / -10.3% |
| **seed floor** (two identical-config sigma=0.05 models, seed only difference) | **-22.9 / -22.5 / -22.2%** |

The residual effect is less than half the seed gap. Nothing is attributable to
noise-training.

### 2. D95 PTV70 at sigma=0.05 is the only survivor — and it is not established

Effect (SmoothAdv vs matched control): **-8.9 / -9.9 / -10.9%** at R/sigma =
0.5 / 1.0 / 1.4. Seed floor: **+4.1 / +4.9 / +5.0%**.

Consistent in sign and magnitude across all three radii, ~2x the floor. BUT the
floor is estimated from a SINGLE pair of models — one difference is not a
standard deviation. Needs 3-5 seeds per arm before it can be claimed.

### 3. sigma=0.02 noise-training is actively harmful (robust)

D95 PTV70 **+7.4 / +8.0 / +7.9%** vs the matched control, frac<=1Gy -1.5 / -3.5 /
-7.1%. Survives both the control and the seed floor.

### 4. frac<=1Gy is not a result

+25.6% at R/sigma=0.5 but +1.5% and -2.7% at 1.0 and 1.4. Not a pattern.

## Clean across-sigma characterisation (no confound — publishable as description)

Same config, sigma the only variable, compared at matched R/sigma:

* **Width scales sublinearly in sigma.** At R/sigma=0.5, sigma 0.05 -> 0.10
  doubles the certified radius for only 1.48x the D95 width (0.8507 -> 1.2612).
  sigma 0.02 -> 0.05 is 2.5x the radius for 1.76x the width.
* **The OAR advantage is a small-sigma phenomenon.** Brainstem/D95 width ratio
  goes 0.64 -> 0.79 -> 0.99 across sigma 0.02/0.05/0.10; by sigma=0.10 the
  brainstem certificate is no tighter than the PTV's.
* **Coverage collapses with sigma.** frac<=1Gy at R/sigma=0.5:
  0.9327 -> 0.5537 -> 0.2953.

## Two errors in RUNPOD_CLAUDE_SMOOTHADV.md (still uncorrected upstream)

1. Step-2 `certify_smoothing.py` omits `--radii`, defaulting to 0.5/1.0/2.0 where
   the certificate is already vacuous (`all_certified: false`) — zero overlap with
   the informative baseline radii.
2. Models are written to the ABSOLUTE `/workspace/results/...`
   (`runpod_train.py:147`), not `./results/`. A relative glob silently matches
   nothing.

## Not run (cut for time)

n=500 tightening at sigma=0.02 on 10 patients, SmoothAdv + control. Would lift
R_max from 1.45*sigma to 2.17*sigma. Orthogonal to every confound above — it
extends the radius grid, it does not strengthen any claim. Worth doing only after
the multi-seed work below.

## Recommended next step

3-5 seeds per arm at sigma=0.05 (SmoothAdv and control). Everything above is
gated on a floor measured from one model pair; that is the single cheapest change
that would turn "suggestive" into a defensible result.
