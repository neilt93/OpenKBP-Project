"""Inject pre-generated perturbed-CT datasets into the training group (robustness retraining).

Goal: train the dose predictor to output the CORRECT (clean) dose even when the CT is
corrupted by one of the robustness perturbation families (P1 noise, P2 bone-shift,
P3 bias-field, P4 resolution, P5 dental). So each injected training sample is:

    CT  = perturbed CT          (from openkbp_hn_robustness/data_perturbed/...)
    dose, masks, possible_dose_mask, voxel_dimensions = the ORIGINAL clean patient

The DataLoader treats a patient as a directory of CSVs keyed by filename, so we just
compose a directory whose `ct.csv` points at the perturbed CT and whose every other file
is symlinked from the matching original patient. No loader changes needed; pass the
composed dirs as extra training patient paths.

LAYOUT NOTE (verify with --inspect when the warehouse is mounted): the perturbed sets
were produced by the robustness pipeline; this script does NOT assume one fixed layout.
Point --perturbed-root at the tree and give a --glob that ends in the per-sample CT file,
with `{pid}` marking where the patient id appears, e.g.:
    "*/*/{pid}/ct.csv"      (data_perturbed/<family>/<level>/pt_3/ct.csv)
    "{pid}/*/ct.csv"        (data_perturbed/pt_3/<family_level>/ct.csv)
If a perturbed patient dir already contains dose + masks, --reuse-existing uses it as-is
instead of composing.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

_PID_RE = re.compile(r"(pt_\d+)")


def _patient_id(path: Path) -> Optional[str]:
    m = _PID_RE.search(str(path))
    return m.group(1) if m else None


def _variant_tag(ct_path: Path, perturbed_root: Path, pid: str) -> str:
    """A unique, filename-safe tag for this perturbed variant (family/level/etc.)."""
    rel = ct_path.parent.relative_to(perturbed_root)
    parts = [p for p in rel.parts if p != pid]
    return "_".join(parts) if parts else "perturbed"


def compose_injected_patient(original_dir: Path, perturbed_ct: Path, out_dir: Path) -> Path:
    """Create out_dir with ct.csv -> perturbed CT and every other file symlinked from
    the original clean patient. Returns out_dir."""
    original_dir, perturbed_ct, out_dir = Path(original_dir), Path(perturbed_ct), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # everything except ct from the original (dose, ROIs, possible_dose_mask, voxel_dims)
    for f in sorted(original_dir.iterdir()):
        if f.name == "ct.csv":
            continue
        link = out_dir / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())
    # ct from the perturbed set
    ct_link = out_dir / "ct.csv"
    if ct_link.is_symlink() or ct_link.exists():
        ct_link.unlink()
    ct_link.symlink_to(perturbed_ct.resolve())
    return out_dir


def build_injected_set(
    original_root: Path,
    perturbed_root: Path,
    out_root: Path,
    glob: str = "*/*/{pid}/ct.csv",
    families: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    max_per_patient: Optional[int] = None,
    holdout_ids: Optional[set] = None,
    reuse_existing: bool = False,
) -> List[Path]:
    """Compose injected patient dirs for every perturbed CT found. Returns the list.

    `glob` locates per-sample CT files under perturbed_root; `{pid}` is replaced by `*`
    for globbing and used to recover the patient id from the matched path.
    `families` / `levels`: keep only variants whose tag contains one of these substrings.
    `max_per_patient`: cap injected variants per clean patient (controls clean:perturbed
        ratio so a few hundred clean patients aren't swamped by thousands of perturbed).
    `holdout_ids`: patient ids that must NEVER be injected (the validation/test split).
        LEAKAGE GUARD: the robustness perturbation sets were generated for the robustness
        study, which evaluated on the *held-out validation patients*. Injecting those into
        training leaks the test set and invalidates every score. Held-out ids are always
        skipped here regardless of `original_root`, and if nothing matches the training
        split we RAISE rather than silently injecting 0 (so you don't "fix" it by pointing
        original_root at validation-pats). If the sets are the validation split, regenerate
        perturbations on the TRAINING CTs instead.
    """
    original_root, perturbed_root, out_root = Path(original_root), Path(perturbed_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    holdout_ids = set(holdout_ids or [])
    pattern = glob.replace("{pid}", "*")
    composed: List[Path] = []
    per_patient: dict = {}
    n_holdout_skipped = n_orphan = 0
    for ct_path in sorted(perturbed_root.glob(pattern)):
        pid = _patient_id(ct_path)
        if pid is None:
            continue
        if pid in holdout_ids:
            n_holdout_skipped += 1          # never inject the held-out split (leakage)
            continue
        original_dir = original_root / pid
        if not (original_dir / "ct.csv").exists():
            n_orphan += 1                   # no matching clean training patient
            continue
        tag = _variant_tag(ct_path, perturbed_root, pid)
        if families and not any(fam in tag for fam in families):
            continue
        if levels and not any(lv in tag for lv in levels):
            continue
        if max_per_patient is not None and per_patient.get(pid, 0) >= max_per_patient:
            continue
        per_patient[pid] = per_patient.get(pid, 0) + 1
        # Unique out-dir name is REQUIRED: the DataLoader keys patients by path.stem
        # (data_loader.py: _patient_to_idx[path.stem]=idx). If two paths share a stem the
        # later silently overwrites the earlier, so every clean pt_X + all its perturbed
        # variants would collapse to one volume. f"{pid}__{tag}" keeps stems unique.
        out_dir = out_root / f"{pid}__{tag}"
        if reuse_existing and (ct_path.parent / "dose.csv").exists():
            # perturbed dir already has dose+masks (e.g. symlinked from the original):
            # symlink them all into the uniquely-named out_dir (do NOT append the raw
            # perturbed dir — its stem is just pid and would collide as above).
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(ct_path.parent.iterdir()):
                link = out_dir / f.name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(f.resolve())
            composed.append(out_dir)
            continue
        composed.append(compose_injected_patient(original_dir, ct_path, out_dir))

    if not composed:
        raise RuntimeError(
            f"Injected 0 perturbed patients (held-out-skipped={n_holdout_skipped}, "
            f"orphan/no-clean-match={n_orphan}). The perturbed sets likely use the "
            f"VALIDATION split (pt_201-240), not training. Verify with --inspect. Do NOT "
            f"point --original-root at validation-pats — that leaks the test set. "
            f"Regenerate perturbations on the TRAINING CTs instead."
        )
    if n_holdout_skipped:
        print(f"  [leakage guard] skipped {n_holdout_skipped} perturbed CTs whose id is in the held-out set")
    return composed


def inspect_layout(perturbed_root: Path, max_show: int = 25) -> None:
    """Print a sample of the perturbed tree so the right --glob can be chosen."""
    perturbed_root = Path(perturbed_root)
    cts = sorted(perturbed_root.rglob("ct.csv"))
    print(f"perturbed root: {perturbed_root}")
    print(f"found {len(cts)} ct.csv files; sample relative paths:")
    for p in cts[:max_show]:
        print(f"  {p.relative_to(perturbed_root)}")
    if cts:
        ex = cts[0].parent
        print(f"\nfirst perturbed dir contents ({ex.relative_to(perturbed_root)}):")
        for f in sorted(ex.iterdir()):
            print(f"  {f.name}")
        print("  -> if dose.csv + ROIs are present here, use --reuse-existing")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject perturbed-CT sets into the training group")
    ap.add_argument("--perturbed-root", required=True)
    ap.add_argument("--inspect", action="store_true", help="Just print the layout and exit")
    ap.add_argument("--original-root", help="Clean patient dirs (e.g. provided-data/train-pats)")
    ap.add_argument("--out-root", help="Where to write composed patient dirs (on the warehouse)")
    ap.add_argument("--glob", default="*/*/{pid}/ct.csv", help="CT glob under perturbed-root; {pid} marks the patient id")
    ap.add_argument("--families", nargs="*", default=None, help="Keep only variants whose tag contains these (e.g. P2 P4)")
    ap.add_argument("--levels", nargs="*", default=None, help="Keep only variants whose tag contains these (e.g. L3 L4)")
    ap.add_argument("--max-per-patient", type=int, default=None, help="Cap injected variants per clean patient (ratio control)")
    ap.add_argument("--holdout-root", default=None, help="Validation/test patient dir; its ids are NEVER injected (leakage guard)")
    ap.add_argument("--reuse-existing", action="store_true", help="Use perturbed dirs directly if they already have dose+masks")
    args = ap.parse_args()

    if args.inspect:
        inspect_layout(Path(args.perturbed_root))
        return
    if not (args.original_root and args.out_root):
        ap.error("--original-root and --out-root are required unless --inspect")
    holdout = None
    if args.holdout_root:
        holdout = {p.name for p in Path(args.holdout_root).iterdir() if _patient_id(p)}
    dirs = build_injected_set(
        Path(args.original_root), Path(args.perturbed_root), Path(args.out_root),
        glob=args.glob, families=args.families, levels=args.levels,
        max_per_patient=args.max_per_patient, holdout_ids=holdout,
        reuse_existing=args.reuse_existing,
    )
    print(f"composed {len(dirs)} injected patient dirs under {args.out_root}")


if __name__ == "__main__":
    main()
