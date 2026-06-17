#!/usr/bin/env python3
"""Step 1: build a matRad input .mat from one (or many) OpenKBP patient dirs.

Run from open-kbp-modified/:
    python openkbp_hn_proton/build_case.py --patient provided-data/validation-pats/pt_201
    python openkbp_hn_proton/build_case.py --patient-range 201 240 \
        --data-root provided-data/validation-pats

Output: openkbp_hn_proton/matrad_cases/<pid>_input.mat (one per patient).
Then run the matRad driver (openkbp_hn_proton/matrad/run_plan.m) where matRad lives.
"""
import argparse
import sys
from pathlib import Path

# Allow `from openkbp_hn_proton ...` whether run as script or module.
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent  # open-kbp-modified/
sys.path.insert(0, str(project_root))

from openkbp_hn_proton.ct_to_matrad import write_matrad_input  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenKBP -> matRad input builder")
    ap.add_argument("--patient", type=str, default=None,
                    help="Path to a single patient dir (e.g. provided-data/validation-pats/pt_201)")
    ap.add_argument("--patient-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                    help="Inclusive pt_<n> range, used with --data-root")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Directory holding pt_<n> dirs (for --patient-range)")
    ap.add_argument("--out-dir", type=str, default=str(script_dir / "matrad_cases"),
                    help="Where to write <pid>_input.mat files")
    ap.add_argument("--no-qc", action="store_true", help="Skip the HU QC printout")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    patient_dirs = []
    if args.patient:
        patient_dirs.append(Path(args.patient))
    elif args.patient_range and args.data_root:
        root = Path(args.data_root)
        for n in range(args.patient_range[0], args.patient_range[1] + 1):
            d = root / f"pt_{n}"
            if d.exists():
                patient_dirs.append(d)
            else:
                print(f"  skip pt_{n}: not found at {d}")
    else:
        ap.error("Provide either --patient or (--patient-range START END --data-root ROOT)")

    if not patient_dirs:
        ap.error("No patient directories resolved. Is the SanDisk warehouse mounted?")

    for d in patient_dirs:
        if not (d / "ct.csv").exists():
            print(f"  ERROR {d}: ct.csv missing (warehouse unmounted?). Skipping.")
            continue
        print(f"\n=== {d.stem} ===")
        out = write_matrad_input(d, out_dir / f"{d.stem}_input.mat", run_qc=not args.no_qc)
        print(f"  wrote {out}")

    print(f"\nDone. Next: run matRad on these in {out_dir} "
          f"via openkbp_hn_proton/matrad/run_plan.m")


if __name__ == "__main__":
    main()
