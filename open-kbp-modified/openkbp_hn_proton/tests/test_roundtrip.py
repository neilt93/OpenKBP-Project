#!/usr/bin/env python3
"""Self-contained Phase-1 tests: HU correction, sparse-CSV round-trip, .mat
assembly, and dose re-import. Needs NO external data and NO matRad.

Run from open-kbp-modified/:
    python openkbp_hn_proton/tests/test_roundtrip.py
    # or: pytest openkbp_hn_proton/tests/test_roundtrip.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

project_root = Path(__file__).resolve().parents[2]  # open-kbp-modified/
sys.path.insert(0, str(project_root))


def _real_load_file():
    """Import the REAL provided_code.utils.load_file in isolation.

    The provided_code package __init__ imports TensorFlow (cluster-only), but
    utils.py itself has no such dependency, so we load it standalone to test our
    output against the actual OpenKBP loader rather than our own reader.
    """
    p = project_root / "provided_code" / "utils.py"
    spec = importlib.util.spec_from_file_location("okbp_utils_isolated", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_file

from openkbp_hn_proton import config as C  # noqa: E402
from openkbp_hn_proton import ct_to_matrad as bridge  # noqa: E402
from openkbp_hn_proton import dose_io  # noqa: E402

SHAPE = C.VOLUME_SHAPE


def _write_sparse(path: Path, indices, data):
    pd.DataFrame(data=data, index=indices, columns=["data"]).to_csv(path)


def _write_mask(path: Path, indices):
    # mask CSVs carry NaN in the data column (the loader keys off that)
    pd.DataFrame(data=np.full(len(indices), np.nan), index=indices, columns=["data"]).to_csv(path)


def make_synthetic_patient(d: Path):
    """Write a tiny but format-faithful OpenKBP patient dir."""
    rng = np.random.default_rng(0)
    n = int(np.prod(SHAPE))

    # Stored CT: a block of "soft tissue" ~1024, an air pocket ~20, some bone ~2000.
    cube = np.zeros(n)
    body = np.arange(1_000_000, 1_400_000)          # arbitrary contiguous body region
    cube[body] = 1024 + rng.normal(0, 30, body.size)  # soft tissue ~ water+0
    air = np.arange(1_100_000, 1_120_000)
    cube[air] = 20.0                                  # internal air cavity
    bone = np.arange(1_300_000, 1_310_000)
    cube[bone] = 2000.0
    cube = np.clip(cube, 0, 4095)
    nz = np.where(cube > 0)[0]
    _write_sparse(d / "ct.csv", nz, cube[nz])

    # Structures: PTV70 (target) and LeftParotid (soft-tissue OAR) inside the body.
    _write_mask(d / "PTV70.csv", np.arange(1_200_000, 1_205_000))
    _write_mask(d / "LeftParotid.csv", np.arange(1_050_000, 1_055_000))

    np.savetxt(d / "voxel_dimensions.csv", np.array([5.422, 5.422, 3.0]))
    return body, air, bone


def test_correct_hu():
    stored = np.full(SHAPE, 0.0)
    scanned = np.zeros(SHAPE, dtype=bool)
    scanned[10:20, 10:20, 10:20] = True
    stored[scanned] = 1024.0          # water-equivalent soft tissue
    true_hu = bridge.correct_hu(stored, scanned)
    assert np.allclose(true_hu[scanned], 0.0, atol=1e-9), "soft tissue should map to ~0 HU"
    assert np.allclose(true_hu[~scanned], C.AIR_HU), "outside scanned region must be air"
    # air cavity stored ~20 -> ~ -1000
    stored[scanned] = 20.0
    assert bridge.correct_hu(stored, scanned)[scanned].max() <= -1000 + 5
    print("PASS test_correct_hu")


def test_sparse_roundtrip():
    rng = np.random.default_rng(1)
    cube = np.zeros(SHAPE)
    idx = rng.choice(int(np.prod(SHAPE)), size=50_000, replace=False)
    cube.ravel()[idx] = rng.uniform(0.1, 75.0, idx.size)  # dose-like, all > 0
    with tempfile.TemporaryDirectory() as t:
        assert dose_io.roundtrip_ok(cube, Path(t) / "dose.csv"), "sparse CSV round-trip lost data"
    print("PASS test_sparse_roundtrip")


def test_build_matrad_input():
    with tempfile.TemporaryDirectory() as t:
        pdir = Path(t) / "pt_999"
        pdir.mkdir()
        make_synthetic_patient(pdir)

        out = bridge.write_matrad_input(pdir, Path(t) / "pt_999_input.mat", run_qc=True)
        md = loadmat(str(out))

        assert tuple(md["cubeHU"].shape) == SHAPE
        assert np.allclose(md["resolution"].ravel(), [5.422, 5.422, 3.0])
        # masks/structures present
        assert "PTV70" in md["masks"].dtype.names
        assert "LeftParotid" in md["masks"].dtype.names
        # HU correction applied: parotid (soft tissue ~1024 stored) median near 0
        parotid = md["masks"]["LeftParotid"][0, 0].astype(bool)
        assert abs(np.median(md["cubeHU"][parotid])) < 60, "soft-tissue HU should be ~0 after correction"
        # air outside the body is -1000
        assert md["cubeHU"].min() <= -999
    print("PASS test_build_matrad_input")


def test_import_matrad_dose():
    # Fake a matRad result .mat with an RBExDose cube, then import it.
    rng = np.random.default_rng(2)
    dose = np.zeros(SHAPE)
    region = (slice(40, 80), slice(40, 80), slice(40, 80))
    dose[region] = rng.uniform(10, 72, dose[region].shape)
    with tempfile.TemporaryDirectory() as t:
        res = Path(t) / "pt_999_dose.mat"
        savemat(str(res), {"RBExDose": dose, "physicalDose": dose / C.RBE})
        out_csv = Path(t) / "dose.csv"
        dose_io.import_matrad_dose(res, out_csv)
        recovered = dose_io.load_sparse_cube(out_csv)
        assert np.allclose(recovered, np.where(dose > 0, dose, 0.0), atol=1e-6)
    print("PASS test_import_matrad_dose")


def test_provided_code_compat():
    """Our sparse output must be readable by the REAL OpenKBP loader, not just ours.

    Catches format bugs (index dtype, index.name, header, NaN handling) that a
    self-round-trip cannot, since save and load would share the same mistake.
    """
    load_file = _real_load_file()
    rng = np.random.default_rng(3)
    cube = np.zeros(SHAPE)
    idx = rng.choice(int(np.prod(SHAPE)), size=20_000, replace=False)
    cube.ravel()[idx] = rng.uniform(0.1, 70.0, idx.size)

    with tempfile.TemporaryDirectory() as t:
        # (a) dose CSV -> real load_file returns a sparse dict that reconstructs exactly
        dose_csv = dose_io.save_sparse_csv(cube, Path(t) / "dose.csv")
        loaded = load_file(dose_csv)
        assert isinstance(loaded, dict) and set(loaded) == {"indices", "data"}, \
            "real loader did not treat our dose CSV as a sparse matrix"
        recon = np.zeros(int(np.prod(SHAPE)))
        recon[loaded["indices"]] = loaded["data"]
        assert np.allclose(recon.reshape(SHAPE), np.where(cube > 0, cube, 0.0)), \
            "real loader reconstructed different values from our dose CSV"

        # (b) a mask CSV (NaN data) -> real load_file returns the raw index array
        mask_idx = np.arange(500_000, 505_000)
        _write_mask(Path(t) / "PTV70.csv", mask_idx)
        mloaded = load_file(Path(t) / "PTV70.csv")
        assert not isinstance(mloaded, dict) and np.array_equal(np.sort(mloaded), mask_idx), \
            "real loader did not treat our mask CSV as a mask"
    print("PASS test_provided_code_compat")


def main():
    test_correct_hu()
    test_sparse_roundtrip()
    test_build_matrad_input()
    test_import_matrad_dose()
    test_provided_code_compat()
    print("\nALL PHASE-1 TESTS PASSED")


if __name__ == "__main__":
    main()
