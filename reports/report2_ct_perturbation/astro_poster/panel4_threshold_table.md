# Panel 4 — Threshold table (criterion: mean shift on any DVH criterion > 1.0 Gy)

| Family | Sweep range tested | Clinically visible at | vs. clinical range | Verdict |
|---|---|---|---|---|
| P1 noise | L1 (8/12 HU) → L5 (100/160 HU) | never in tested range (max 0.14 Gy) | typical scanner noise ≈10–50 HU (≈L1–L2) | Robust |
| P2 HU shift | L1 (5/50 HU) → L5 (100/1000 HU) | never in tested range (max 0.61 Gy) | inter-scanner calibration drift ≈10–50 HU (≈L2) | Robust |
| P3 bias field | L1 (10 HU) → L5 (200 HU) | never in tested range (max 0.09 Gy) | RF/scatter non-uniformity, tens of HU | Robust |
| P4 resolution | L0 (0.5/0.25 vox) → L4 (4.0/2.0 vox) | L2 (2.0/1.0): 1.15 Gy on Larynx_D_0.1_cc | cross-scanner slice/kernel variation ≈1–3 vox (≈L1–L3) | SENSITIVE — the failure mode |
| P5 dental streak | L1 (150HU/8) → L5 (1200/24) | never in tested range (max 0.33 Gy) | dental streaks common in H&N (L2–L4 realistic) | Robust |
