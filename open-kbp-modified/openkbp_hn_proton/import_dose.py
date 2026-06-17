#!/usr/bin/env python3
"""Step 3: import matRad result .mat -> OpenKBP sparse proton-dose CSV.

Run from open-kbp-modified/:
    python openkbp_hn_proton/import_dose.py \
        --result openkbp_hn_proton/matrad_cases/pt_201_dose.mat \
        --out    openkbp_hn_proton/proton_dose/pt_201/dose.csv

The output dose.csv is drop-in proton ground truth in OpenKBP format (Gy(RBE)),
ready to symlink alongside the original CT/structure files for training.
"""
import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from openkbp_hn_proton.dose_io import import_matrad_dose  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="matRad dose .mat -> OpenKBP sparse CSV")
    ap.add_argument("--result", required=True, help="Path to matRad result .mat")
    ap.add_argument("--out", required=True, help="Output dose CSV path")
    ap.add_argument("--field", default=None,
                    help="Explicit dose field name (else auto: rbeDose/RBExDose/physicalDose/dose)")
    args = ap.parse_args()

    out = import_matrad_dose(Path(args.result), Path(args.out), field=args.field)
    print(f"Wrote proton dose CSV: {out}")


if __name__ == "__main__":
    main()
