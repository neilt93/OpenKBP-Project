#!/usr/bin/env python3
"""Step 1: Generate perturbed CT data for robustness evaluation.

Creates perturbed copies of validation patient CT volumes using 5 perturbation
families at 2 severity levels each. Runs locally (no GPU needed).

Usage:
    python -m openkbp_hn_robustness.generate_perturbed_data [--config CONFIG] [--verify-only]
    # Or from open-kbp-modified/:
    python openkbp_hn_robustness/generate_perturbed_data.py [--config CONFIG] [--verify-only]
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure parent directory is on path for direct script execution
script_dir = Path(__file__).parent.resolve()
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

import numpy as np
import yaml

from openkbp_hn_robustness.perturbations import (
    ALL_PERTURBATIONS,
    load_ct_volume,
    create_perturbed_patient,
    load_structure_mask,
)
from openkbp_hn_robustness.perturbations.base import VOLUME_SHAPE

logger = logging.getLogger(__name__)


def make_seed(global_seed: int, perturbation_name: str, level: str, patient_id: str) -> int:
    """Generate a deterministic seed from the combination of parameters."""
    key = f"{global_seed}_{perturbation_name}_{level}_{patient_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def compute_psnr(original: np.ndarray, perturbed: np.ndarray, body_mask: np.ndarray) -> float:
    """Compute PSNR between original and perturbed volumes within body mask."""
    body_orig = original[body_mask]
    body_pert = perturbed[body_mask]
    mse = np.mean((body_orig - body_pert) ** 2)
    if mse == 0:
        return float('inf')
    max_val = body_orig.max()
    return 10.0 * np.log10(max_val**2 / mse)


def verify_patient(perturbed_dir: Path, original_dir: Path) -> dict:
    """Verify a perturbed patient directory for QC."""
    ct_path = perturbed_dir / "ct.csv"
    if not ct_path.exists():
        return {"error": "ct.csv not found"}

    perturbed_vol, p_mask = load_ct_volume(perturbed_dir)
    original_vol, o_mask = load_ct_volume(original_dir)

    body_voxels = perturbed_vol[p_mask]
    orig_body = original_vol[o_mask]

    return {
        "mean_hu": float(body_voxels.mean()),
        "std_hu": float(body_voxels.std()),
        "min_hu": float(body_voxels.min()),
        "max_hu": float(body_voxels.max()),
        "n_body_voxels": int(p_mask.sum()),
        "n_orig_body_voxels": int(o_mask.sum()),
        "psnr": compute_psnr(original_vol, perturbed_vol, o_mask),
        "clip_fraction": float(np.mean(
            (perturbed_vol[o_mask] <= 1.0) | (perturbed_vol[o_mask] >= 4095.0)
        )),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate perturbed CT data")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data, don't regenerate")
    parser.add_argument("--perturbations", nargs="+", default=None, help="Only run specific perturbations (e.g., P1_noise P3_bias_field)")
    parser.add_argument("--levels", nargs="+", default=None, help="Only run specific levels (e.g., L1 L2)")
    parser.add_argument("--patients", nargs="+", default=None, help="Only run specific patients (e.g., pt_201 pt_205)")
    args = parser.parse_args()

    # Determine project root (open-kbp-modified/)
    project_root = script_dir.parent

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir / "configs" / "default.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_seed = config["seed"]

    # Set up paths
    validation_dir = project_root / config["paths"]["validation_data"]
    output_root = project_root / config["paths"]["data_perturbed"]

    # Set up logging
    log_dir = project_root / config["paths"]["output_root"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "generation.log"),
            logging.StreamHandler(),
        ],
    )

    # Build patient list
    if args.patients:
        patient_ids = args.patients
    else:
        start = config["patients"]["start"]
        end = config["patients"]["end"]
        patient_ids = [f"pt_{i}" for i in range(start, end + 1)]

    # Filter to existing patients
    patient_dirs = {}
    for pid in patient_ids:
        pdir = validation_dir / pid
        if pdir.exists():
            patient_dirs[pid] = pdir
        else:
            logger.warning(f"Patient directory not found: {pdir}")

    logger.info(f"Found {len(patient_dirs)} patients")

    # Instantiate perturbations
    perturbation_instances = {p.name: p() for p in ALL_PERTURBATIONS}

    # Filter perturbations if specified
    if args.perturbations:
        perturbation_instances = {
            k: v for k, v in perturbation_instances.items()
            if k in args.perturbations
        }

    # QC summary
    qc_summary = {}

    for p_name, perturbation in perturbation_instances.items():
        levels_to_run = args.levels if args.levels else list(perturbation.levels.keys())

        for level in levels_to_run:
            if level not in perturbation.levels:
                logger.warning(f"Level {level} not defined for {p_name}, skipping")
                continue

            condition_key = f"{p_name}/{level}"
            logger.info(f"{'Verifying' if args.verify_only else 'Generating'} {condition_key} "
                        f"({len(patient_dirs)} patients)")

            condition_stats = []

            for pid, src_dir in patient_dirs.items():
                dst_dir = output_root / p_name / level / pid

                if args.verify_only:
                    if dst_dir.exists():
                        stats = verify_patient(dst_dir, src_dir)
                        stats["patient"] = pid
                        condition_stats.append(stats)
                        logger.info(f"  {pid}: mean={stats.get('mean_hu', 'N/A'):.1f} "
                                    f"std={stats.get('std_hu', 'N/A'):.1f} "
                                    f"PSNR={stats.get('psnr', 'N/A'):.1f} "
                                    f"clip={stats.get('clip_fraction', 'N/A'):.4f}")
                    else:
                        logger.warning(f"  {pid}: directory not found")
                    continue

                # Load CT volume
                ct_volume, body_mask = load_ct_volume(src_dir)

                # Prepare kwargs (mandible mask for P5)
                kwargs = {}
                if p_name == "P5_dental":
                    kwargs["mandible_mask"] = load_structure_mask(src_dir, "Mandible")
                    kwargs["patient_dir"] = src_dir

                # Generate deterministic seed and RNG
                seed = make_seed(global_seed, p_name, level, pid)
                rng = np.random.default_rng(seed)

                # Apply perturbation
                perturbed = perturbation.apply(ct_volume, body_mask, level, rng, **kwargs)

                # Save perturbed patient
                create_perturbed_patient(src_dir, dst_dir, perturbed)

                # QC stats
                psnr = compute_psnr(ct_volume, perturbed, body_mask)
                body_pert = perturbed[body_mask]
                clip_frac = float(np.mean((body_pert <= 1.0) | (body_pert >= 4095.0)))

                stats = {
                    "patient": pid,
                    "mean_hu": float(body_pert.mean()),
                    "std_hu": float(body_pert.std()),
                    "psnr": psnr,
                    "clip_fraction": clip_frac,
                }
                condition_stats.append(stats)

                logger.info(f"  {pid}: mean={stats['mean_hu']:.1f} std={stats['std_hu']:.1f} "
                            f"PSNR={psnr:.1f} clip={clip_frac:.4f}")

            qc_summary[condition_key] = condition_stats

    # Save QC summary
    qc_path = output_root / "qc_summary.json"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qc_path, "w") as f:
        json.dump(qc_summary, f, indent=2)
    logger.info(f"QC summary saved to {qc_path}")

    # Print aggregate stats
    print(f"\n{'=' * 70}")
    print(f"{'Condition':<25} {'Mean HU':>10} {'Std HU':>10} {'PSNR':>10} {'Clip %':>10}")
    print(f"{'=' * 70}")
    for condition, stats_list in qc_summary.items():
        if not stats_list:
            continue
        mean_hu = np.mean([s["mean_hu"] for s in stats_list])
        std_hu = np.mean([s["std_hu"] for s in stats_list])
        mean_psnr = np.mean([s["psnr"] for s in stats_list if s["psnr"] != float('inf')])
        mean_clip = np.mean([s["clip_fraction"] for s in stats_list])
        print(f"{condition:<25} {mean_hu:>10.1f} {std_hu:>10.1f} {mean_psnr:>10.1f} {mean_clip:>10.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
