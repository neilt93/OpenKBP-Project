# Decision rules — pre-committed BEFORE the seed grid is run

Written and committed before any Phase 2 run exists, so the grid's verdict cannot be reinterpreted
after the numbers land. The two load-bearing disciplines: **one difference per claim**, and **rules
written before results**. If a later analysis wants to depart from these rules, it must say so
explicitly and justify it — not silently.

## Arms (identical fixed pipeline, `--aug-path numpy`, paired seeds 1–3)

| Arm | Config | Isolates |
|-----|--------|----------|
| **A** | numpy path, `--aug-noise 0` | reference (pipeline effect vs frozen baseline) |
| **B** | numpy path, `--aug-noise 0.05` | noise-training effect (B vs A) |
| **C** | frozen restored baseline (no training) | "did continued training itself do it" (A vs C) |
| **D** | numpy path + PGD adv-training (2–3 steps, ε=0.02) | adversarial-training effect (D vs A) |

"Paired seed" = the same seed is compared across arms so seed luck cancels in the difference.

## What counts as a real effect

An effect (a per-metric width change between two arms) is declared **real** only if it satisfies
**both** gates:

1. **Magnitude gate.** The seed-averaged effect exceeds **2× the paired-seed spread** of the
   relevant metric, where the spread is the std across the 3 paired seeds (not a single pair).
2. **Accuracy gate.** The arm's clean prediction accuracy is within tolerance of Arm A's:
   **DVH score within +0.10 and Dose score within +0.15** of Arm A (absolute, higher = worse).
   A narrower certificate from a model that failed the accuracy gate is recorded as
   **"tighter but less accurate" (a flatness artifact), not a robustness win.**

Both gates must pass. A width change that clears magnitude but fails accuracy is **not** a result;
neither is one that holds accuracy but is within the seed floor.

## Statistics

- Paired **Wilcoxon signed-rank** across the 40 patients on seed-averaged per-patient widths, for
  every declared effect. Report the p-value; treat p < 0.05 as significant (descriptive, not a
  multiplicity-corrected family claim unless stated).
- Report both **mean** and **worst-case-patient** widths.
- Every certificate number is printed **next to its clean-accuracy number**. No width is reported
  without its accuracy.

## Escape hatch (decided in advance)

If the paired-seed spread on D95|PTV70 comes back **> 10%**, extend arms A and B to **5 seeds**
before concluding anything. If it is tight (**~5%**), **3 seeds** suffice.

## Pre-registered expectations (so we can be wrong on the record)

- σ=0.02 noise-training is expected **harmful** on D95|PTV70 (prior: +8% vs control, survived both
  controls). σ=0.02 is therefore **not** in the grid.
- The σ=0.05 D95|PTV70 ~−10% "survivor" is **expected to fail the accuracy gate or the magnitude
  gate** once seeds and accuracy are in — it is the thing this grid is built to kill or confirm.
- The A-vs-C "pipeline effect" (~25% on brainstem) is expected to be **real but is a property of
  the augmentation pipeline, not of noise-training** — it must not be reported as a SmoothAdv win.

## Any outcome is a result

The grid is designed so that every outcome is publishable: "plain continued training is the lever
(A≈B, A≠C)," "noise adds a small real σ=0.05 effect (B<A, both gates)," "adversarial training buys
local robustness at X accuracy cost (D)," or "nothing beats the seed floor." The verdict comes from
these rules, not from the vibe of the table.
