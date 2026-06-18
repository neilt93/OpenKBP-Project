"""Tests for provided_code.inject_perturbed (path composition; no TF/data needed).

Run:  python tests/test_inject.py
"""
import importlib.util
import tempfile
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "provided_code" / "inject_perturbed.py"
_spec = importlib.util.spec_from_file_location("inject_perturbed", _p)
_inj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inj)


def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_compose_and_build():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        # clean original patient
        orig = root / "train-pats" / "pt_3"
        _write(orig / "ct.csv", "CLEAN_CT")
        _write(orig / "dose.csv", "DOSE")
        _write(orig / "PTV70.csv", "MASK")
        _write(orig / "possible_dose_mask.csv", "PDM")
        _write(orig / "voxel_dimensions.csv", "VOX")
        # perturbed CTs: data_perturbed/<family>/<level>/pt_3/ct.csv
        _write(root / "data_perturbed" / "P2_boneshift" / "L4" / "pt_3" / "ct.csv", "PERTURBED_P2")
        _write(root / "data_perturbed" / "P4_resolution" / "L4" / "pt_3" / "ct.csv", "PERTURBED_P4")
        # a perturbed id with no matching clean patient -> must be skipped
        _write(root / "data_perturbed" / "P2_boneshift" / "L4" / "pt_999" / "ct.csv", "ORPHAN")

        out = root / "injected"
        dirs = _inj.build_injected_set(
            root / "train-pats", root / "data_perturbed", out, glob="*/*/{pid}/ct.csv"
        )
        assert len(dirs) == 2, f"expected 2 composed (orphan skipped), got {len(dirs)}: {dirs}"

        # check the P2 composed dir: ct -> perturbed, everything else -> original
        d = out / "pt_3__P2_boneshift_L4"
        assert d in dirs
        assert (d / "ct.csv").read_text() == "PERTURBED_P2", "ct must be the perturbed CT"
        assert (d / "dose.csv").read_text() == "DOSE", "dose must be the clean ground truth"
        assert (d / "PTV70.csv").read_text() == "MASK"
        assert (d / "possible_dose_mask.csv").read_text() == "PDM"
        assert (d / "voxel_dimensions.csv").read_text() == "VOX"
        assert (d / "ct.csv").is_symlink() and (d / "dose.csv").is_symlink()
        print("PASS test_compose_and_build (ct=perturbed, dose/masks=clean, orphan skipped)")


def test_family_filter():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        orig = root / "train-pats" / "pt_1"
        for f in ("ct.csv", "dose.csv", "PTV70.csv"):
            _write(orig / f, f)
        for fam in ("P1_noise", "P2_boneshift", "P4_resolution"):
            _write(root / "dp" / fam / "L5" / "pt_1" / "ct.csv", fam)
        dirs = _inj.build_injected_set(
            root / "train-pats", root / "dp", root / "out",
            glob="*/*/{pid}/ct.csv", families=["P2", "P4"],
        )
        tags = sorted(d.name for d in dirs)
        assert all("P1" not in x for x in tags), tags
        assert len(dirs) == 2, tags
        print(f"PASS test_family_filter (kept {tags})")


def test_leakage_guard_skips_holdout():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        for pid in ("pt_1", "pt_201"):  # pt_201 = held-out validation
            _write(root / "train-pats" / pid / "ct.csv", "ct")
            _write(root / "train-pats" / pid / "dose.csv", "dose")
            _write(root / "dp" / "P2" / "L4" / pid / "ct.csv", f"pert_{pid}")
        dirs = _inj.build_injected_set(
            root / "train-pats", root / "dp", root / "out",
            glob="*/*/{pid}/ct.csv", holdout_ids={"pt_201"},
        )
        names = [d.name for d in dirs]
        assert all("pt_201" not in n for n in names), names  # held-out never injected
        assert any("pt_1" in n for n in names)
        print("PASS test_leakage_guard_skips_holdout (pt_201 excluded from training)")


def test_raises_when_all_holdout():
    # the realistic failure: perturbed sets ARE the validation split -> must raise, not 0
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        _write(root / "train-pats" / "pt_201" / "ct.csv", "ct")
        _write(root / "dp" / "P2" / "L4" / "pt_201" / "ct.csv", "pert")
        try:
            _inj.build_injected_set(root / "train-pats", root / "dp", root / "out",
                                    glob="*/*/{pid}/ct.csv", holdout_ids={"pt_201"})
            raise AssertionError("should have raised on 0 injected")
        except RuntimeError as e:
            assert "VALIDATION" in str(e)
            print("PASS test_raises_when_all_holdout (loud failure, no silent-0 / no leak)")


def test_levels_and_cap():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        _write(root / "train-pats" / "pt_1" / "ct.csv", "ct")
        _write(root / "train-pats" / "pt_1" / "dose.csv", "dose")
        for fam in ("P2", "P4"):
            for lv in ("L3", "L4", "L5"):
                _write(root / "dp" / fam / lv / "pt_1" / "ct.csv", f"{fam}{lv}")
        # levels filter
        d_lv = _inj.build_injected_set(root / "train-pats", root / "dp", root / "o1",
                                       glob="*/*/{pid}/ct.csv", levels=["L4"])
        assert len(d_lv) == 2 and all("L4" in d.name for d in d_lv), [d.name for d in d_lv]
        # cap per patient
        d_cap = _inj.build_injected_set(root / "train-pats", root / "dp", root / "o2",
                                        glob="*/*/{pid}/ct.csv", max_per_patient=2)
        assert len(d_cap) == 2, [d.name for d in d_cap]
        print("PASS test_levels_and_cap (level filter + per-patient cap)")


if __name__ == "__main__":
    test_compose_and_build()
    test_family_filter()
    test_leakage_guard_skips_holdout()
    test_raises_when_all_holdout()
    test_levels_and_cap()
    print("\nALL INJECTION TESTS PASSED")
