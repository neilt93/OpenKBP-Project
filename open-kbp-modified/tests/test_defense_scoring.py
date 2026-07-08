"""Tests for provided_code.defense_scoring (numpy + DoseEvaluator, NO TF).

The valuable one is test_score_prediction_*: it drives the REAL DoseEvaluator
through synthetic data via score_prediction, proving the in-memory DVH path
(no prediction CSVs) gives correct competition scores. The rest pin the
bookkeeping arithmetic that produces the headline numbers (clean cost, fraction
of attack damage recovered).

Run:  python tests/test_defense_scoring.py
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # open-kbp-modified

# Register a fake provided_code package so submodule imports (batch, data_loader,
# dose_evaluation_class) resolve WITHOUT running provided_code/__init__.py, which
# pulls in TensorFlow. Mirrors openkbp_hn_robustness/evaluate_metrics.py.
if "provided_code" not in sys.modules:
    _pkg = types.ModuleType("provided_code")
    _pkg.__path__ = [str(ROOT / "provided_code")]
    sys.modules["provided_code"] = _pkg

_spec = importlib.util.spec_from_file_location("defense_scoring", ROOT / "provided_code" / "defense_scoring.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

from provided_code.dose_evaluation_class import DoseEvaluator  # noqa: E402  (fake pkg)

ROIS = ["Brainstem", "PTV70"]


class _FakeLoader:
    rois = {"oars": ["Brainstem"], "targets": ["PTV70"]}
    full_roi_list = ROIS
    patient_id_list = ["pt_1"]


class _FakeBatch:
    def __init__(self, masks, voxdim, pdm, pid="pt_1"):
        self.structure_masks = masks
        self.voxel_dimensions = voxdim
        self.possible_dose_mask = pdm
        self.patient_list = [pid]

    def get_index_structure_from_structure(self, name):
        return ROIS.index(name)


def _synthetic_batch(D=8):
    masks = np.zeros((1, D, D, D, 2), dtype=np.uint8)
    masks[0, 1:4, 1:4, 1:4, 0] = 1   # Brainstem cube
    masks[0, 4:7, 4:7, 4:7, 1] = 1   # PTV70 cube
    pdm = np.ones((1, D, D, D, 1), dtype=np.uint8)  # whole volume deliverable
    voxdim = np.array([5.4, 5.4, 3.0])
    return _FakeBatch(masks, voxdim, pdm)


def test_score_prediction_identical_is_zero():
    ev = DoseEvaluator(_FakeLoader())
    batch = _synthetic_batch()
    ref = np.full(8 ** 3, 30.0)            # uniform 30 Gy
    ds.score_prediction(ev, batch, ref, ref.copy())
    dose_score, dvh_score = ev.get_scores()
    assert abs(dose_score) < 1e-9 and abs(dvh_score) < 1e-9, (dose_score, dvh_score)
    print("PASS test_score_prediction_identical_is_zero")


def test_score_prediction_unit_shift():
    ev = DoseEvaluator(_FakeLoader())
    batch = _synthetic_batch()
    ref = np.full(8 ** 3, 30.0)
    pred = ref + 1.0                       # +1 Gy everywhere (pdm covers all)
    ds.score_prediction(ev, batch, ref, pred)
    dose_score, dvh_score = ev.get_scores()
    # Every DVH metric (D_99/95/1, mean, D_0.1_cc) shifts by exactly 1 Gy; dose MAE = 1.
    assert abs(dose_score - 1.0) < 1e-6, dose_score
    assert abs(dvh_score - 1.0) < 1e-6, dvh_score
    print("PASS test_score_prediction_unit_shift")


def test_build_defense_list():
    full = ds.build_defense_list(["none", "smooth", "noise"], quick=False)
    assert full == [("none", None), ("smooth", 0.5), ("smooth", 1.0), ("smooth", 2.0),
                    ("noise", 0.02), ("noise", 0.05), ("noise", 0.10)], full
    quick = ds.build_defense_list(["none", "smooth", "noise"], quick=True)
    assert quick == [("none", None), ("smooth", 1.0), ("noise", 0.05)], quick
    print("PASS test_build_defense_list")


def test_cond_key():
    assert ds.cond_key("fgsm", 0.05, "smooth", 1.0) == "fgsm|eps0.05|smooth|1"
    assert ds.cond_key("clean", 0.0, "none", None) == "clean|eps0|none|"
    print("PASS test_cond_key")


def test_add_derived_math():
    recs = [
        {"attack": "clean", "epsilon": 0.0, "defense": "none", "strength": None,
         "dvh_score": 2.0, "dose_score": 3.0},
        {"attack": "clean", "epsilon": 0.0, "defense": "smooth", "strength": 1.0,
         "dvh_score": 2.1, "dose_score": 3.05},
        {"attack": "fgsm", "epsilon": 0.05, "defense": "none", "strength": None,
         "dvh_score": 3.0, "dose_score": 4.0},
        {"attack": "fgsm", "epsilon": 0.05, "defense": "smooth", "strength": 1.0,
         "dvh_score": 2.2, "dose_score": 3.2},
    ]
    ds.add_derived(recs)
    clean_smooth = recs[1]
    fgsm_none = recs[2]
    fgsm_smooth = recs[3]
    assert abs(clean_smooth["clean_cost_dvh"] - 0.1) < 1e-9
    # Undefended under attack recovers nothing.
    assert abs(fgsm_none["recovered_frac_dvh"] - 0.0) < 1e-9
    # smooth: damage 1.0, remaining 0.2 -> 80% recovered; clean cost 0.1.
    assert abs(fgsm_smooth["recovered_frac_dvh"] - 0.8) < 1e-9, fgsm_smooth
    assert abs(fgsm_smooth["recovered_frac_dose"] - 0.8) < 1e-9, fgsm_smooth
    assert abs(fgsm_smooth["clean_cost_dvh"] - 0.1) < 1e-9
    print("PASS test_add_derived_math")


if __name__ == "__main__":
    test_score_prediction_identical_is_zero()
    test_score_prediction_unit_shift()
    test_build_defense_list()
    test_cond_key()
    test_add_derived_math()
    print("\nAll defense_scoring tests passed.")
