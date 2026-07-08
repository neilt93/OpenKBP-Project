# Proton Dose-Prediction Robustness — Research Plan

Reference doc for the proton direction. Reframes the work from "predict proton dose" (crowded,
and undercut by auto-plan quality) to a specific, verified-open niche that reuses everything we
have already built for photons.

## Context and motivation
We already have, for photons: a robustness evaluation toolchain (FGSM/PGD adversarial + clinically
realistic CT perturbations), a training-free test-time defence (CT noise, recovers 60–100% of
attack damage), and adversarial/injection retraining that hardened the model. We also have a
validated pipeline that GENERATES proton IMPT ground truth on the public OpenKBP H&N anatomy with
matRad (no MATLAB; end-to-end on pt_201). The strategic question is how to turn that into a novel,
defensible paper rather than an incremental "proton U-Net" result.

Physics gives the hook: proton range depends directly on CT → stopping-power along the beam, so a
proton dose predictor should be MORE sensitive to CT input error than a photon one. Range
uncertainty is the central clinical problem in proton therapy. That makes protons the natural
stress case for a study of *model-reliability* robustness.

## The novelty gap (deep literature scan, 2026-07-08, 25/25 claims verified)
The intersection **"model-reliability perturbation robustness of a deep-learning PROTON dose/fluence
predictor + a proposed mitigation + a matched photon-vs-proton contrast"** is unclaimed as of early
2026. Every nearest neighbour misses at least one axis. What is NOT novel (state this plainly):
proton dose prediction itself; proton *plan/statistical* robustness (scenario/percentile doses);
and a photon-vs-proton predictor contrast for *accuracy*. The open wedge is the combination.

### Related work and exactly what each misses
| Work | Modality | Robustness type | Mitigation? | Matched contrast? | Misses |
|---|---|---|---|---|---|
| Gao, Mody, Rao, Staring 2025 (PMB, 10.1088/1361-6560/adcfeb) | Photon | Model-reliability (adversarial + noise) | No (eval only) | No | proton, mitigation |
| Mgboh et al. 2026, arXiv 2605.00904 (Wayne State/Henry Ford, AIME) | Photon IMRT (fluence) | Clinically-realistic shifts, "not adversarial" | Ambiguous; at most a physics-informed loss + SwinUNETR rec, never adversarial/transform/smoothing/ensembling | No | proton, adversarial, clear mitigation |
| Base FluenceFormer, arXiv 2511.08645 | Photon IMRT | none (accuracy) | No | No | robustness entirely |
| Vazquez 2023 (PMB accc08); Liang 2024 (ad780b); Nomer 2024 (ad8c95) | Proton IMPT | **Plan/statistical** (scenario/percentile dose) — the EXCLUDED sense | No | No | model-reliability robustness, mitigation |
| Guerreiro et al. 2021 (Radiother Oncol, S0167814020311968) | Photon + proton, matched | none (accuracy, modality selection) | No | **Yes (accuracy)** | any robustness |
| arXiv 2606.30115 (2026) | CT (detection) | Adversarial (FGSM) | **Yes (dynamic adversarial training, 75%→7%)** | No | dose prediction, proton |
| DoseRAD2026, arXiv 2604.12778 (public, 115 pts, photon+proton) | Photon + proton | none | No | n/a | it is dose-CALCULATION (MC surrogate), not KBP dose-PREDICTION, no robustness track |

Take-away: the proton robustness papers are all plan-robustness (excluded); the model-reliability
robustness papers are photon-only and mostly evaluation-focused; the matched photon-vs-proton
predictor exists only for accuracy; the one CT adversarial+mitigation paper is a detection task; and
the emerging public proton benchmark is a different task (calculation, not prediction).

## Our contributions (the intersection nobody occupies)
1. **Matched-anatomy photon-vs-proton model-reliability robustness contrast.** Same OpenKBP patients,
   two dose targets (photon from OpenKBP, proton from our matRad pipeline). Run the same FGSM/PGD +
   clinical CT-perturbation suites on both predictors. **This controlled contrast is the headline and
   the novelty** — Guerreiro did the matched contrast for accuracy, never for robustness.
2. **Mitigation transferred to proton, quantified against photon.** Apply the test-time noise defence
   and adversarial retraining (both already built) to the proton model; report how much reliability
   each recovers, and whether protons benefit more (bigger problem) or less (harder problem). No prior
   proton predictor has ANY model-reliability mitigation.
3. **Reusable methodology.** The matRad-on-OpenKBP proton ground-truth generation pipeline (Octave,
   no MATLAB), released with honest auto-plan-quality caveats — a route to proton DL research without
   proprietary data, complementary to DoseRAD2026 (which is calculation, not prediction).

## Hypotheses (physically grounded)
- **H1:** the proton predictor degrades MORE than the matched photon predictor under the same CT
  perturbation, because proton range tracks CT → stopping-power. This is the physically-motivated,
  testable core claim.
- **H2:** the mitigations still help for protons, but the recovery profile differs from photon.
- A **null** H1 (proton not more fragile) is itself interesting and publishable — it would suggest the
  learned predictor does not "see" range the way a physics engine does, which is a finding in its own
  right. Either outcome is a result.

## Experimental plan (most infrastructure already prepped)
- **Phase 2 — generate proton dose** (CPU x86 box). `batch_generate.sh` for train (1–200) + validation
  (201–240); assembles an OpenKBP-format proton set. (Ready.)
- **Phase 3 — train the proton predictor** (GPU). `runpod_train.py --data-dir proton-data`, photon-
  winning recipe (SE + aug + DVH loss 0.02). Photon fixes already merged (`b4bf56b`). (Ready.)
- **Phase 4 — the robustness contrast.** Run the existing adversarial + CT-perturbation evaluation on
  BOTH the photon and proton predictors, same patients. Primary output: degradation curves,
  photon vs proton, per attack / perturbation family.
- **Phase 5 — mitigation.** Apply test-time defence + retraining to the proton model; recovery vs the
  photon case.

### Headline figures
1. **Photon-vs-proton degradation curves** on matched anatomy (the contrast).
2. **Physics tie-in:** a CT perturbation → induced proton range shift → predictor error, connecting the
   ML result to the mechanism (photon-only papers cannot show this).
3. **Recovery under mitigation**, photon vs proton.

## Why this survives our data caveat
Every robustness number is degradation from each model's OWN clean baseline. So even though the matRad
auto-plans have known quality limits (PTV edge coverage gaps, parotid hotspots; no clinical sign-off),
the *relative* fragility-and-recovery story is unaffected. This is precisely why we can proceed without
clinical plan tuning and still have a defensible paper — an accuracy claim could not.

## Risks and caveats (respect these in print)
- **2605.00904 mitigation ambiguity.** Verifiers disagreed on whether its physics-informed loss counts
  as a robustness mitigation. Read its Methods/Conclusion in full before asserting "evaluation-only" in
  print; either way it is photon and has no adversarial/transform/smoothing/ensembling defence.
- **DoseRAD2026** could add a prediction-from-contours track or a robustness follow-up. Track it; it is
  also a potential external proton data source. Re-check arXiv listings right before submission.
- **Absence of evidence.** The novelty claim is "no such work found in a verified corpus," not a proof
  of nonexistence. A very recent or non-indexed preprint could exist.
- **Ground-truth quality.** Spot-check generated proton dose across patients (QC step in the RunPod
  brief), not just pt_201.
- **Effect size.** H1 could be weaker than the physics suggests; plan the paper so a null is still a
  contribution.

## Target venue
A Medical Physics / Physics in Medicine & Biology paper, positioned as the mitigation + matched-contrast
follow-up to Gao et al. 2025 and Mgboh et al. 2026. Complements the in-flight SERA 2026 (photon
adversarial) and ASTRO 2026 (photon perturbation) submissions — this is the proton extension of that
line of work.

## Status and next steps
- **Ready (off-box):** `proton-pipeline` pushed; `batch_generate.sh`; `RUNPOD_CLAUDE_PROTON.md`; photon
  fixes merged (`b4bf56b`); `--data-dir` wired; all tests pass.
- **Needs boxes:** Phase 2 on a native x86 CPU box, then Phases 3–5 on a 4090.
- **Immediate:** run Phase 2. The novelty is confirmed, so the compute is now the bottleneck, not the
  positioning.

## Key references (verified)
- Gao, Mody, Rao, Staring 2025, PMB 70(11) 115006, 10.1088/1361-6560/adcfeb — photon robustness, eval only.
- Mgboh, Sultan, Kim, Thind, Zhu 2026, arXiv 2605.00904 — photon fluence robustness (clinically realistic).
- Vazquez 2023 (PMB accc08); Liang 2024 (ad780b); Nomer 2024 (ad8c95) — proton PLAN robustness.
- Guerreiro et al. 2021, Radiother Oncol S0167814020311968 — matched photon/proton predictor (accuracy).
- arXiv 2606.30115 2026 — CT adversarial robustness + dynamic adversarial training (detection task).
- DoseRAD2026, arXiv 2604.12778 — public photon+proton dose-CALCULATION challenge.
- Athalye et al. 2018 — input-transform defences fall to adaptive attacks (our honest defence caveat).
