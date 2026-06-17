# OpenKBP H&N Proton Pipeline (Phase 1)

Generate **proton IMPT ground-truth dose** on the OpenKBP H&N anatomy with matRad,
then train the existing 3D U-Net to predict proton dose. OpenKBP ships only photon
dose, so the proton dose must be *generated* — that generation is the new work; the
network, data loader, sparse-CSV format, and DVH/dose scoring are reused unchanged.

## Pipeline

```
OpenKBP patient dir ──build_case.py──▶ <pid>_input.mat ──run_plan.m (matRad)──▶ <pid>_dose.mat ──import_dose.py──▶ dose.csv (OpenKBP sparse, Gy(RBE))
   (CT + masks)         [Python]          (ct cube + cst)      [MATLAB/Octave]      (RBExDose cube)      [Python]        drop-in proton ground truth
```

| Step | Command (run from `open-kbp-modified/`) | Where |
|---|---|---|
| 1. Build matRad input | `python openkbp_hn_proton/build_case.py --patient provided-data/validation-pats/pt_201` | local / cluster |
| 2. Optimize IMPT plan | `matlab -batch "addpath(genpath('<matRad>')); run('openkbp_hn_proton/matrad/run_plan.m')"` | matRad env |
| 3. Import dose | `python openkbp_hn_proton/import_dose.py --result openkbp_hn_proton/matrad_cases/pt_201_dose.mat --out openkbp_hn_proton/proton_dose/pt_201/dose.csv` | local / cluster |

`run_plan.m` is also callable as `run_plan(inputDir, outputDir)` and batch-processes
every `*_input.mat` in `matrad_cases/`, skipping cases that already have a dose.

## The vertical slice (do this before any 240-patient batch)

Prove the chain on **one** patient. The format/logic is verified now; the physical
plan needs your environment.

| Check | Status |
|---|---|
| HU correction (soft tissue → ~0, air → −1000) | ✅ verified on synthetic data (`tests/test_roundtrip.py`) |
| Sparse-CSV dose round-trip (cube → csv → cube, exact) | ✅ verified |
| `.mat` assembly + dose re-import | ✅ verified |
| Output readable by the **real** `provided_code` loader | ✅ verified (`test_provided_code_compat` exercises the actual `load_file`) |
| Real `pt_201` HU QC | ✅ verified on real data (2026-06-15): soft tissue 19 HU, parotids −15/−18, air at −1000 (91.6% of vol). HU_OFFSET=1024 correct. |
| Real `pt_201` spatial machinery + slice axis | ✅ verified (photon dry-run): dose hugs PTV on all 3 planes; **axis2 = S-I through-slice** (3.0 mm), in-plane axes isotropic 5.422 mm. No transpose/flip bug. |
| matRad runs end-to-end (Octave/Docker) + dose **spatially aligned** on pt_201 | ✅ verified (2026-06-17): full stf → particle dose calc → IPOPT optimize → import. Dose is conformal — hugs PTV on all 3 planes. Geometry/index chain confirmed correct (mask `.mat` roundtrip exact; dose-CSV roundtrip exact). |
| Plan **quality** (PTV70 D95 ≥ ~95% of 70, OARs under limit) | ⏳ partial — PTV70 ~90% covered at ~70 Gy but ~10% exact-zero coverage gaps (5 mm spots) + small disjoint PTV63/56 under-covered + parotid hotspots. Needs 3 mm spots (native x86/RunPod) and objective tuning; see below. |

> The gate is *spatial alignment*, not just DVH. D95 is computed against the same mask
> matRad used, so a geometrically-wrong plan (bad slice axis / beam direction) can still
> score fine. `qc_proton_dose.py` renders a CT+PTV+dose overlay precisely to catch that.

Run the format tests now: `python openkbp_hn_proton/tests/test_roundtrip.py`

When the warehouse is mounted (before matRad), confirm HU offset + spatial machinery —
you can do this on the **existing photon dose** already:
```bash
python openkbp_hn_proton/build_case.py --patient provided-data/validation-pats/pt_201
#   -> read the "HU QC" block: parotid/soft-tissue median must be near 0 HU, not ~1000

python openkbp_hn_proton/qc_proton_dose.py \
    --patient provided-data/validation-pats/pt_201 \
    --dose    provided-data/validation-pats/pt_201/dose.csv   # photon dose, as a dry run
#   -> confirm the overlay's high-dose region sits on the PTV and identify the slice axis
```

After matRad makes proton dose, run the **real gate** on it:
```bash
python openkbp_hn_proton/qc_proton_dose.py \
    --patient provided-data/validation-pats/pt_201 \
    --dose    openkbp_hn_proton/proton_dose/pt_201/dose.csv
#   -> spatial overlay + DVH through the real DoseEvaluator; PTV70 D95 should be ~>= 95% of 70
```

## Running matRad — no MATLAB needed (Octave + Docker)

matRad runs on free **GNU Octave**. Ubuntu 22.04 ships Octave 6.4.0, which matches
matRad's precompiled solver `thirdParty/IPOPT/ipopt.mexoct640a64` exactly (statically
linked → no compilation). The `matrad/` dir has everything:

```bash
# one-time: clone matRad (kept off iCloud, on the warehouse) + build the image
git clone https://github.com/e0404/matRad.git "<warehouse>/tools/matRad"
cp "<warehouse>/tools/matRad/thirdParty/IPOPT/ipopt.mexoct640a64" \
   "<warehouse>/tools/matRad/thirdParty/IPOPT/ipopt.mex"   # activate the linux-octave mex
docker build --platform linux/amd64 -t openkbp-matrad:octave640 -f matrad/Dockerfile matrad/

# per batch: optimize every *_input.mat in matrad_cases/ -> *_dose.mat
MATRAD_DIR="<warehouse>/tools/matRad" matrad/docker_octave.sh matrad_cases
```

`docker_octave.sh` mounts matRad + the cases dir and runs `matrad/run_entry.m`
(`matRad_rc` → `disableGUI` → `run_plan`). `run_plan.m` mirrors this matRad version's
`example5_protons` flow: `pln.bioModel='constRBE'`, `multScen='nomScen'`,
`engine='HongPB'`, `quantityOpt='RBExDose'`, fine 3 mm dose grid interpolated back to
the 128³ CT grid, output scaled ×`numOfFractions` to total-course Gy(RBE).

> **amd64 emulation note (Apple Silicon):** the x86 solver runs under emulation —
> fine for one-patient validation (~4 min at 5 mm spots), but 3 mm spots (~3× the
> variables) exceed the 7.6 GB emulated container. Run the **240-patient batch on a
> native x86_64 box (RunPod)** with the same image, where 3 mm spots fit and run fast.

### Plan-quality tuning (the open Phase-1 item)
The pipeline is correct end-to-end; the **auto-plan quality** still needs work before
it's good ground truth. Quantified on pt_201 with **3 mm spots / 4 mm dose grid**
(112k variables, fits the 7.6 GB local container — ~6 min):
- **PTV70**: D50 = 69 Gy (median at prescription), D2 = 73 (no big hotspot), the dose is
  conformal — BUT D95 ≈ 3 and only ~60% of the volume gets ≥ 90% of prescription. The
  under-dose is at the **target's S-I edges** (most superior/inferior slices) and the
  periphery — a penumbra/margin + beam-coverage effect, not a bug.
- **Small disjoint PTV63/PTV56 largely unreached** (D50 ≈ 18 / 12; ~10% / 3% coverage).
  matRad's objectives are per-structure-normalized, so this is **reachability**, not
  voxel-count weighting — the single-isocenter 3-beam axial arrangement doesn't serve
  the separate small volumes.
- **Parotid hotspots** (Dmax ~75 ≫ 26 limit) — the OAR-vs-target objective balance is
  off; needs stronger OAR penalty / max-dose objectives or DVH objectives.
- **Ruled out** (not the cause of under-coverage): lateral spot density (3 mm barely
  changed it), dose-grid sampling, orientation/index (roundtrips exact), objective
  voxel-count weighting (normalized).
- **Diagnosis:** the open issues are **beam geometry + planning protocol** (margins,
  beam angles per disjoint target, objective balance) — clinical-judgment tuning, not a
  pipeline defect. This is where **Birjoo's input** (beam arrangement, prescription, OAR
  constraints) is the right next step, alongside adding PTV-margin / DVH objectives.

**Local compute note:** 3 mm spots fit the 7.6 GB Docker container only by pairing them
with a 4 mm (not 3 mm) dose grid so the dij memory stays ~constant. On a bigger box use
3 mm spots + 3 mm grid. matRad dose calc is CPU-only — no GPU involved at any point.

## Design decisions (in `config.py` — held FIXED across all patients)

KBP only works if anatomy→dose is ~deterministic, and IMPT dose is strongly
beam-dependent, so these are a deliberate planning *protocol*, not free parameters:

- **HU offset = 1024** — `trueHU = stored − 1024`. Verified: OpenKBP soft tissue sits
  at ~1019 stored → ~0 HU; air dropped to sparse-zero → set to −1000. Re-confirm per
  dataset via the QC printout.
- **Fixed beam template** — gantry `[180, 60, 300]`, couch `[0,0,0]` (posterior + two
  anterior obliques, bilateral H&N). *Revisit after reviewing Phase-1 plan quality.*
  In-plane axes (axis0/axis1, both 5.422 mm) are isotropic, so resolution is exact, but
  which is anatomical L-R vs A-P isn't pinned down — gantry angles are defined in matRad's
  frame, so the template may be rotated relative to true anatomy. Harmless for a *fixed*
  protocol (valid dose, identical for every patient → still learnable); confirm beam entry
  directions look anatomically sane on the pt_201 matRad overlay and adjust angles if not.
- **SIB prescription** — PTV70/63/56 → 70/63/56 Gy(RBE), 35 fractions.
- **Objectives** — squared-deviation on targets, squared-overdosing on OARs at standard
  H&N limits. Same weights for everyone.
- **Constant RBE 1.1**, output **Gy(RBE)** to stay consistent with the `/70` training
  normalization. Nominal-scenario (not robust) optimization for v1.

## Known limitations / watch-items

- **Coarse grid:** OpenKBP voxels are 5.422 × 5.422 × 3.0 mm — very coarse laterally for
  proton range/penumbra. Fine for a proof-of-concept predictor; name it as a limitation.
- **Plan quality gate:** a single fixed objective template will under-serve hard
  anatomies, and the network only learns as well as the *worst* training plan. Use the
  `run_plan.m` QC printout (PTV70 D95, parotid mean) to flag/triage bad plans.
- **matRad version:** written against matRad v3 (object-based `DoseObjectives`). The
  dose-calc call is guarded for `matRad_calcParticleDose` vs `matRad_calcDoseInfluence`.
- **Photon-vs-proton comparison is confounded** (OpenKBP photon = human plans vs proton
  = auto-plans) — keep it a secondary observation, not a headline.

## Once proton dose exists

`proton_dose/<pid>/dose.csv` is OpenKBP-format Gy(RBE). Symlink it alongside the original
CT/structure files (or point a new training dir at it) and retrain with the **unchanged**
`runpod_train.py` — same inputs, same `/70` normalization, same scoring. Store generated
dose on the SanDisk warehouse, not in the iCloud project dir.
