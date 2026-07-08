# RunPod Claude Brief — Proton IMPT ground truth (Phase 2) + retrain (Phase 3)

You are a Claude Code instance on a RunPod box. Goal: generate proton IMPT ground-truth dose
for the OpenKBP H&N patients with matRad (Phase 2), then retrain the 3D U-Net to predict proton
dose (Phase 3). Phase 1 (the OpenKBP↔matRad bridge) is done and validated on pt_201; see
`openkbp_hn_proton/README.md` for the design. This branch is `proton-pipeline`.

> **Two different boxes.** Phase 2 (matRad) is **CPU-only on native x86** — no GPU, no CUDA.
> Phase 3 (training) is a **GPU** box (RTX 4090, like the photon runs). Do NOT run Phase 2 on
> Apple-Silicon emulation: 3 mm spots exceed the emulated container. Provision a high-vCPU x86
> CPU instance for Phase 2, then a 4090 for Phase 3.

> **Plan-quality caveat (accepted).** The auto-generated plans have known limits (PTV70 D95
> coverage gaps at the S-I edges, under-covered small disjoint PTV63/56, parotid hotspots — see
> README). We are proceeding WITHOUT clinical sign-off, so the proton ground truth carries these
> limitations and the U-Net will learn them. That is an accepted trade-off for a first end-to-end
> proton model, not a bug. Note it in any write-up.

---

## PHASE 2 — generate proton dose (CPU, native x86)

### 2.0 Box
- High-vCPU **x86_64 CPU** instance (e.g. 32+ vCPU, 32+ GB RAM). No GPU needed.
- Docker available. Disk: ~30 GB (matRad + OpenKBP data + per-patient `.mat`/dose).

### 2.1 Code + data + matRad
```bash
cd /workspace
git clone https://github.com/neilt93/OpenKBP-Project.git openkbp && cd openkbp
git checkout proton-pipeline
pip install numpy scipy pandas tqdm more_itertools
cd open-kbp-modified
# Public OpenKBP data on-box: sparse-checkout train-pats (pt_1-200) + validation-pats
# (pt_201-240) into provided-data/  (skip test-pats unless you also want a proton test set).

# matRad (no MATLAB — Octave in Docker). Clone, activate the linux-octave ipopt mex, build image.
export MATRAD_DIR=/workspace/matRad
git clone https://github.com/e0404/matRad.git "$MATRAD_DIR"
cp "$MATRAD_DIR/thirdParty/IPOPT/ipopt.mexoct640a64" "$MATRAD_DIR/thirdParty/IPOPT/ipopt.mex"
docker build --platform linux/amd64 -t openkbp-matrad:octave640 \
    -f openkbp_hn_proton/matrad/Dockerfile openkbp_hn_proton/matrad/
```

### 2.2 Smoke-test on ONE patient first (matches the validated pt_201 run)
```bash
python openkbp_hn_proton/build_case.py --patient provided-data/validation-pats/pt_201
MATRAD_DIR=$MATRAD_DIR openkbp_hn_proton/matrad/docker_octave.sh openkbp_hn_proton/matrad_cases
python openkbp_hn_proton/import_dose.py \
    --result openkbp_hn_proton/matrad_cases/pt_201_dose.mat \
    --out openkbp_hn_proton/proton_dose/pt_201/dose.csv
python openkbp_hn_proton/qc_proton_dose.py --patient provided-data/validation-pats/pt_201 \
    --dose openkbp_hn_proton/proton_dose/pt_201/dose.csv    # dose must hug the PTV on all 3 planes
```
If the QC overlay is conformal, proceed to the batch.

### 2.3 Batch all 240 (train + validation)
`batch_generate.sh <data_root> <start> <end> <proton_data_out>` builds inputs, runs matRad
(skips finished cases), imports each dose, and assembles an OpenKBP-format proton dir
(OpenKBP CT+masks symlinked, proton dose as `dose.csv`). Re-runnable — resumes on re-invoke.
```bash
export MATRAD_DIR=/workspace/matRad
openkbp_hn_proton/matrad/batch_generate.sh provided-data/train-pats      1   200 proton-data/train-pats
openkbp_hn_proton/matrad/batch_generate.sh provided-data/validation-pats 201 240 proton-data/validation-pats
```
- **This is the long step:** ~6 min/patient at 3 mm spots, so ~24 h for 240 sequential. **Parallelise
  by RANGE:** split into K chunks and run K invocations concurrently, each with its OWN matRad copy
  (`cp -r $MATRAD_DIR $MATRAD_DIR_k`; `docker_octave` mounts matRad read-write, so copies must not be
  shared). Size K to the vCPU count. Watch RAM (each container holds one patient's dij).
- **Persist the outputs** to the network volume / SanDisk: `proton-data/` (the training set),
  `openkbp_hn_proton/proton_dose/` (the dose CSVs), and `matrad_cases/*_dose.mat` (so a re-run skips
  finished patients). These are the expensive artefacts — back them up before tearing the box down.
- Spot-check a few with `qc_proton_dose.py` across the range (not just pt_201) to catch any
  patient-specific geometry issue before training on all 240.

---

## PHASE 3 — retrain the U-Net on proton dose (GPU)

### 3.0 Code is already prepared (no merge needed)
The photon `adversarial-retraining` fixes (the S-I augmentation flip, leakage-guard hardening,
DVH-loss fix, and the `--data-dir` flag) are **already merged into `proton-pipeline`** (merge
commit `b4bf56b`, verified off-box: `test_bugfixes.py`, `test_dvh_percentile.py`, and the proton
`test_roundtrip.py` all pass). A fresh `git checkout proton-pipeline` has everything — just
re-confirm on the box:
```bash
python tests/test_bugfixes.py && python tests/test_dvh_percentile.py && python openkbp_hn_proton/tests/test_roundtrip.py
```

### 3.1 Train (GPU box, TF 2.18.0, same best config as photon)
Point the loader at the assembled proton data with `--data-dir proton-data` (it expects
`proton-data/train-pats` and `proton-data/validation-pats`, which `batch_generate.sh` built).
The network, loader, sparse-CSV format, and DVH/dose scoring are unchanged — only the dose
values are proton now.
```bash
python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug \
    --use-dvh --dvh-weight 0.02 --batch-size 4 --ptv-weight 4.0 --no-jit \
    --data-dir proton-data
```
- Use the photon-winning recipe as the starting point (SE + aug + DVH loss w=0.02, the 1.837
  config), but treat the proton numbers as a fresh baseline — proton dose has different structure,
  so scores are not comparable to the photon DVH 1.837.
- Then a 5-seed ensemble if the single model looks reasonable, same as photon.

### 3.2 Evaluate (Phase 4)
Score with the existing `DoseEvaluator` on the proton validation split, and run the robustness
eval (CT perturbations / adversarial) — robustness is MORE meaningful for protons (range depends
on CT→stopping-power), which is the whole point of the proton direction.

---

## Guardrails
- **Never train on the validation split** (pt_201-240) — Phase 3 trains on `proton-data/train-pats`
  (pt_1-200) only. The leakage guard is hardened but the split is still your responsibility.
- Phase 2 is **x86 CPU**; Phase 3 is **GPU + TF 2.18.0**. Do not conflate the boxes.
- Persist `proton-data/`, `proton_dose/`, and `matrad_cases/*_dose.mat` to the volume/SanDisk —
  regenerating 240 proton plans is ~a day of CPU.

## Report back
- Phase 2: how many of 240 generated cleanly, any patients matRad failed, wall-clock + vCPU used,
  a couple of QC overlays across the range.
- Phase 3: proton single-model + ensemble DVH/Dose (as a fresh proton baseline), final model paths.
