"""Tests for provided_code.gamma_index (pure numpy, NO TF). Run: python tests/test_gamma_index.py"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if "provided_code" not in sys.modules:
    _pkg = types.ModuleType("provided_code")
    _pkg.__path__ = [str(ROOT / "provided_code")]
    sys.modules["provided_code"] = _pkg
_spec = importlib.util.spec_from_file_location("gamma_index", ROOT / "provided_code" / "gamma_index.py")
gi = importlib.util.module_from_spec(_spec)
sys.modules["gamma_index"] = gi
_spec.loader.exec_module(gi)

SP = [3.0, 3.0, 3.0]


def _dose():
    rng = np.random.default_rng(0)
    d = np.zeros((20, 20, 20), np.float32)
    d[5:15, 5:15, 5:15] = 60.0 + rng.normal(0, 1, (10, 10, 10)).astype(np.float32)  # ~60 Gy blob
    return d


def test_identical_is_100pct():
    d = _dose()
    assert gi.gamma_passing_rate(d, d, SP, 3.0, 3.0) == 1.0


def test_small_dose_diff_passes():
    d = _dose()
    ev = d + 0.5  # 0.5 Gy << 3% of ~60 Gy (=1.8 Gy) -> should pass everywhere
    assert gi.gamma_passing_rate(d, ev, SP, 3.0, 3.0) > 0.99


def test_large_dose_diff_fails():
    d = _dose()
    ev = d + 10.0  # 10 Gy >> 1.8 Gy DD and no DTA rescue -> should mostly fail
    assert gi.gamma_passing_rate(d, ev, SP, 3.0, 3.0) < 0.1


def test_small_spatial_shift_passes_via_dta():
    d = _dose()
    ev = np.roll(d, 1, axis=0)  # 1 voxel = 3 mm shift; 3mm DTA should rescue it
    # compare on interior to avoid roll wrap-around artefacts dominating
    gpr = gi.gamma_passing_rate(d, ev, SP, 3.0, 3.0)
    assert gpr > 0.9, gpr


def test_threshold_excludes_low_dose():
    d = _dose()
    # corrupt only low-dose (zero) region; it's below threshold so GPR should stay ~1.
    ev = d.copy(); ev[0:3, 0:3, 0:3] = 50.0
    assert gi.gamma_passing_rate(d, ev, SP, 3.0, 3.0, dose_threshold_frac=0.10) > 0.99


def test_shift_helper_inverse():
    a = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    out = gi._shift_fill(a, (1, 0, 0), fill=np.inf)
    # out[i]=a[i+1]; last plane has no source -> inf
    assert np.array_equal(out[0], a[1])
    assert np.isinf(out[2]).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} gamma-index tests passed.")
