"""Configuration for the OpenKBP -> matRad proton pipeline.

Everything that is NOT patient anatomy lives here and is held FIXED across every
patient. For knowledge-based planning to be learnable, the anatomy->dose mapping
must be (near) deterministic; varying beam angles or objective weights per patient
makes it one-to-many and the network cannot learn it. Treat these as a deliberate
planning protocol, not arbitrary defaults.
"""
from __future__ import annotations

VOLUME_SHAPE = (128, 128, 128)

# --- Axis convention (RESOLVED 2026-06-15 from real masks) -------------------
# OpenKBP cube is C-order numpy; scipy.io.savemat preserves logical indices, so
# numpy axis k -> MATLAB dimension k+1. Verified anatomical mapping (consistent
# across a 12-patient train/val/test sample):
#   axis0 = Anterior-Posterior  (spinal cord at HIGHER index = posterior), 5.422 mm
#   axis1 = Left-Right          (LeftParotid at HIGHER index),             5.422 mm
#   axis2 = Superior-Inferior   (through-slice / stacked axial slices),    3.000 mm
# This matches matRad's expected cube layout dim1=y(A-P), dim2=x(L-R), dim3=z(S-I)
# with NO 90-deg rotation, so resolution = [x_or_y, x_or_y, z]=[5.422,5.422,3.0] is
# exact. Residual to confirm visually on the pt_201 matRad overlay: the gantry-angle
# SIGN convention (does 180 deg enter posterior?) — the template below is symmetric
# in its obliques (60/300), so an L-R flip is harmless; only an A-P sign flip would
# move the single unpaired beam from posterior to anterior.

# --- HU convention -----------------------------------------------------------
# OpenKBP stores RESCALED HU = trueHU + HU_OFFSET, sparse (only stored>0 kept),
# air outside the scanned region dropped to 0. matRad's HU->stopping-power lookup
# expects STANDARD HU (water=0, air=-1000). So: trueHU = stored - HU_OFFSET inside
# the scanned region, AIR_HU everywhere else.
#
# HU_OFFSET default 1024 (12-bit positive shift). VERIFY per dataset with
# ct_to_matrad.qc_hu(): soft tissue should land near 0 HU and air near -1000.
HU_OFFSET = 1024.0
AIR_HU = -1000.0
HU_MIN = -1000.0   # clip floor for matRad hlut
HU_MAX = 3071.0    # clip ceiling for matRad hlut

# --- Structures (mirror provided_code/data_loader.py) ------------------------
OARS = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"]
TARGETS = ["PTV56", "PTV63", "PTV70"]

# --- SIB prescription, Gy(RBE) ----------------------------------------------
# The PTV names already encode the simultaneous-integrated-boost dose levels.
# RESOLVED 2026-06-15: OpenKBP PTV56/63/70 are stored as DISJOINT volumes, not
# nested — verified zero pairwise overlap across a 12-patient train/val/test sample.
# So each voxel gets exactly one squared-deviation prescription; no conflicting
# objectives, no exclusive-shell construction needed. (PTV70 is present for every
# patient; PTV63/PTV56 are sometimes absent — build_matrad_input includes only
# the masks actually contoured.)
PRESCRIPTIONS = {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0}

# --- Fixed IMPT beam template (SAME for every patient) ----------------------
# Posterior + two anterior obliques is a defensible bilateral-H&N arrangement.
# This is a design choice to revisit after reviewing plan quality (Phase 1 QC).
GANTRY_ANGLES = [180.0, 60.0, 300.0]
COUCH_ANGLES = [0.0, 0.0, 0.0]
BIXEL_WIDTH = 5.0  # mm (pencil-beam spot spacing). 5 mm runs reliably under local
                   # Docker x86 emulation (~4 min/patient). NOTE: 5 mm leaves some
                   # exact-zero coverage gaps in the target (~10% of PTV70 voxels);
                   # 3 mm fills them but ~3x the spots — too heavy for the 7.6 GB
                   # emulated container (gets killed mid dose-calc). Use 3 mm for the
                   # 240-patient production batch on native x86 (RunPod), where it fits.

# --- matRad cst objective weights -------------------------------------------
TARGET_PENALTY = 1000.0   # SquaredDeviation on each PTV at its prescription
OAR_PENALTY = 300.0       # SquaredOverdosing on each OAR above its limit
# OAR planning dose limits, Gy(RBE) (standard H&N constraints)
OAR_MAX_DOSE = {
    "Brainstem": 54.0,
    "SpinalCord": 45.0,
    "RightParotid": 26.0,
    "LeftParotid": 26.0,
    "Esophagus": 45.0,
    "Larynx": 45.0,
    "Mandible": 70.0,
}

# --- physics -----------------------------------------------------------------
RADIATION_MODE = "protons"
MACHINE = "Generic"
RBE = 1.1  # constant-RBE fallback if matRad returns only physicalDose

# --- dose normalization (matches DataLoader) --------------------------------
DOSE_PRESCRIPTION = 70.0  # Gy, for /70 normalization at training time


def structure_type(name: str) -> str:
    """Return 'TARGET' or 'OAR' for a structure name."""
    if name in TARGETS:
        return "TARGET"
    if name in OARS:
        return "OAR"
    raise ValueError(f"Unknown structure: {name}")
