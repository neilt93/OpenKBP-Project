# Certified Robustness of Dose Prediction — project folder

Home for the **certified-robustness** study (the newer thread; NOT the ASTRO 2026 CT-perturbation
poster). Turns the test-time CT-noise defense into (a) an adaptive-attack-tested empirical defense
and (b) a provable median-smoothing certificate with clinical DVH guarantees.

## Layout

```
reports/certified_robustness/
├── README.md              # this file
├── PAPER_OUTLINE.md       # purpose / method / results / conclusion skeleton + gaps
├── literature/
│   └── LITERATURE_REVIEW.md   # cited, verified gap analysis (deep-research, 2026-07-31)
├── experiment_results/    # verified result JSONs, full 40-patient validation (from RunPod)
│   ├── adaptive_full/ adaptive_strong/ adaptive_results_eot16/   # §1 EOT adaptive attack
│   ├── certify_s0.02_full/ certify_s0.05_full/ certify_s0.10_full/  # §2 certification σ-sweep
│   ├── certify_n500_validradii_10pat/ certify_s0.10_n500_maxR/   # n=500 tighter/max-radius
│   └── sanity_s1/ sanity_s2/                                     # n=10 sanity passes
└── paper_figures/         # figures for the paper (contour-overlay, σ-tradeoff, etc.)
```

## Code (lives with the codebase, not here)

- `open-kbp-modified/adversarial_adaptive.py` — EOT adaptive attack on the noise defense
- `open-kbp-modified/certify_smoothing.py` — median-smoothing certification driver
- `open-kbp-modified/provided_code/smoothing_certify.py` — certificate math (unit-tested)
- `open-kbp-modified/save_contour_overlay_figures.py` — contour-safety figures
- `open-kbp-modified/RUNPOD_CLAUDE_CERTIFY.md` — run brief

Also mirrored in S3: `s3://5jwj898h77/results/` (results) and `s3://5jwj898h77/models/` (model).

## Status (2026-07-31)

Core results in hand (adaptive attack held; first certified DVH bounds at small radii). Verified
literature gap. Remaining before submission: SmoothAdv run, harden adaptive claim, gamma-index
results, contour figure, a few HU-magnitude citations, the write-up. See PAPER_OUTLINE.md §Gaps.
