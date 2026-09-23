# Dose MAE and DVH error per condition (40 test patients)

Dose MAE (Gy) = voxel-wise mean absolute error within the possible-dose mask (the OpenKBP dose score). DVH error (Gy) = mean absolute error over the 23 DVH criteria (the OpenKBP DVH score). Deltas are vs the unperturbed baseline.

| Family                  | Level   |   Dose MAE (Gy) | Delta MAE (Gy)   |   DVH error (Gy) | Delta DVH (Gy)   | Delta DVH (%)   |
|:------------------------|:--------|----------------:|:-----------------|-----------------:|:-----------------|:----------------|
| Baseline                |         |           3.731 |                  |            2.535 |                  |                 |
| P1 Acquisition noise    | L1      |           3.731 | 0.001            |            2.535 | 0.0              | 0.01            |
| P1 Acquisition noise    | L2      |           3.733 | 0.002            |            2.538 | 0.003            | 0.13            |
| P1 Acquisition noise    | L3      |           3.737 | 0.006            |            2.543 | 0.009            | 0.34            |
| P1 Acquisition noise    | L4      |           3.742 | 0.011            |            2.559 | 0.024            | 0.96            |
| P1 Acquisition noise    | L5      |           3.742 | 0.011            |            2.571 | 0.036            | 1.43            |
| P2 HU calibration shift | L1      |           3.73  | -0.001           |            2.531 | -0.004           | -0.15           |
| P2 HU calibration shift | L2      |           3.74  | 0.01             |            2.544 | 0.01             | 0.38            |
| P2 HU calibration shift | L3      |           3.751 | 0.02             |            2.519 | -0.016           | -0.61           |
| P2 HU calibration shift | L4      |           3.793 | 0.062            |            2.572 | 0.037            | 1.46            |
| P2 HU calibration shift | L5      |           3.916 | 0.186            |            2.818 | 0.284            | 11.19           |
| P3 Bias field           | L1      |           3.732 | 0.001            |            2.535 | 0.0              | 0.01            |
| P3 Bias field           | L2      |           3.73  | -0.0             |            2.534 | -0.001           | -0.04           |
| P3 Bias field           | L3      |           3.736 | 0.006            |            2.534 | -0.0             | -0.01           |
| P3 Bias field           | L4      |           3.733 | 0.002            |            2.524 | -0.011           | -0.44           |
| P3 Bias field           | L5      |           3.759 | 0.028            |            2.545 | 0.011            | 0.42            |
| P4 Resolution loss      | L0      |           3.739 | 0.008            |            2.573 | 0.038            | 1.5             |
| P4 Resolution loss      | L1      |           3.78  | 0.049            |            2.636 | 0.101            | 4.0             |
| P4 Resolution loss      | L2      |           3.885 | 0.154            |            2.729 | 0.194            | 7.67            |
| P4 Resolution loss      | L3      |           3.982 | 0.252            |            2.824 | 0.289            | 11.4            |
| P4 Resolution loss      | L4      |           4.124 | 0.393            |            2.995 | 0.46             | 18.15           |
| P5 Dental artifact      | L1      |           3.729 | -0.002           |            2.541 | 0.007            | 0.26            |
| P5 Dental artifact      | L2      |           3.73  | -0.001           |            2.545 | 0.011            | 0.42            |
| P5 Dental artifact      | L3      |           3.73  | -0.0             |            2.552 | 0.017            | 0.67            |
| P5 Dental artifact      | L4      |           3.731 | 0.0              |            2.557 | 0.022            | 0.87            |
| P5 Dental artifact      | L5      |           3.732 | 0.002            |            2.555 | 0.021            | 0.82            |
