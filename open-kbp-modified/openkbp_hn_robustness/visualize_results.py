#!/usr/bin/env python3
"""Step 4: Generate figures and tables for the robustness analysis.

Creates publication-quality figures for the ASTRO abstract:
1. Summary bar chart — all conditions side by side
2. Heatmap — 5 perturbations x 2 levels for dose/DVH degradation
3. Per-structure radar plot — which OARs/PTVs are most affected
4. Example CT slices — clean vs perturbed for each perturbation
5. Dose difference maps — predicted dose (clean) vs predicted dose (perturbed)

Usage:
    cd open-kbp-modified/
    python openkbp_hn_robustness/visualize_results.py [--config CONFIG]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

PERTURBATION_LABELS = {
    "P1_noise": "Acq. Noise",
    "P2_bone_shift": "Bone Shift",
    "P3_bias_field": "Bias Field",
    "P4_resolution": "Resolution",
    "P5_dental": "Dental Art.",
}

LEVEL_COLORS = {
    "L0": "#77AADD",
    "L1": "#4C72B0",
    "L2": "#DD8452",
    "L3": "#CC6677",
    "L4": "#AA4499",
    "L5": "#882255",
}


def load_results(metrics_dir: Path) -> tuple:
    """Load summary CSV and detailed JSON."""
    summary_df = pd.read_csv(metrics_dir / "summary.csv")
    details_path = metrics_dir / "detailed_results.json"
    with open(details_path) as f:
        details = json.load(f)
    return summary_df, details


def _get_levels_for_perturbation(summary_df: pd.DataFrame, p_name: str) -> list:
    """Extract sorted level names (L0, L1, ...) present in the data for a perturbation."""
    all_levels = set()
    for cond in summary_df["perturbation"].values:
        if cond.startswith(f"{p_name}/"):
            level = cond.split("/")[1]
            all_levels.add(level)
    # Sort: L0, L1, L2, ... (lexicographic works for L0-L9)
    return sorted(all_levels)


def plot_degradation_curves(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Line plot: dose/DVH score vs perturbation level for each perturbation type.

    Shows where degradation takes off (gradual vs sudden) per perturbation.
    Dynamically reads available levels from the data (handles different level
    counts per perturbation, e.g. P4 has L0-L4, others have L1-L5).
    """
    p_names = list(PERTURBATION_LABELS.keys())

    # Get baseline scores
    baseline_row = summary_df[summary_df["perturbation"] == "baseline"]
    if baseline_row.empty:
        print("  Skipping degradation curves (no baseline)")
        return
    baseline_dose = baseline_row["dose_score"].iloc[0]
    baseline_dvh = baseline_row["dvh_score"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set1(np.linspace(0, 0.6, len(p_names)))

    for idx, (metric, baseline_val, ylabel, title) in enumerate([
        ("dose_score", baseline_dose, "Dose Score (Gy)", "Dose Score Degradation"),
        ("dvh_score", baseline_dvh, "DVH Score", "DVH Score Degradation"),
    ]):
        ax = axes[idx]

        for pi, p_name in enumerate(p_names):
            levels = _get_levels_for_perturbation(summary_df, p_name)
            if not levels:
                continue

            x_labels_line = ["Baseline"] + levels
            x_positions_line = list(range(len(x_labels_line)))
            scores = [baseline_val]
            for level in levels:
                condition = f"{p_name}/{level}"
                row = summary_df[summary_df["perturbation"] == condition]
                if not row.empty:
                    scores.append(row[metric].iloc[0])
                else:
                    scores.append(np.nan)

            label = PERTURBATION_LABELS.get(p_name, p_name)
            ax.plot(x_positions_line[:len(scores)], scores, 'o-', color=colors[pi],
                    label=label, linewidth=2, markersize=7)

        # Build unified x-axis from all levels across perturbations
        all_levels_union = sorted({
            lvl for p in p_names
            for lvl in _get_levels_for_perturbation(summary_df, p)
        })
        x_labels_all = ["Baseline"] + all_levels_union
        x_positions_all = list(range(len(x_labels_all)))

        ax.set_xticks(x_positions_all)
        ax.set_xticklabels(x_labels_all)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(figures_dir / "degradation_curves.png", bbox_inches='tight')
    fig.savefig(figures_dir / "degradation_curves.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved degradation_curves.png/pdf")


def plot_summary_bars(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart of dose and DVH scores for all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conditions = summary_df["perturbation"].values
    dose_scores = summary_df["dose_score"].values
    dvh_scores = summary_df["dvh_score"].values

    # Color by condition type
    colors = []
    for c in conditions:
        if c == "baseline":
            colors.append("#2CA02C")
        else:
            level = c.split("/")[-1] if "/" in c else "L1"
            colors.append(LEVEL_COLORS.get(level, "#999999"))

    # Shorten labels
    labels = []
    for c in conditions:
        if c == "baseline":
            labels.append("Baseline")
        else:
            parts = c.split("/")
            p_label = PERTURBATION_LABELS.get(parts[0], parts[0])
            labels.append(f"{p_label}\n{parts[1]}")

    x = np.arange(len(conditions))

    # Dose score
    axes[0].bar(x, dose_scores, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel("Dose Score (Gy)")
    axes[0].set_title("Dose Score by Condition")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    if "baseline" in conditions:
        baseline_dose = summary_df.loc[summary_df["perturbation"] == "baseline", "dose_score"].iloc[0]
        axes[0].axhline(y=baseline_dose, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        axes[0].legend()

    # DVH score
    axes[1].bar(x, dvh_scores, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel("DVH Score")
    axes[1].set_title("DVH Score by Condition")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    if "baseline" in conditions:
        baseline_dvh = summary_df.loc[summary_df["perturbation"] == "baseline", "dvh_score"].iloc[0]
        axes[1].axhline(y=baseline_dvh, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(figures_dir / "summary_bars.png", bbox_inches='tight')
    fig.savefig(figures_dir / "summary_bars.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary_bars.png/pdf")


def plot_degradation_heatmap(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Heatmap: perturbations x levels showing degradation %."""
    p_names = list(PERTURBATION_LABELS.keys())
    # Dynamically collect all levels present across all perturbations
    levels = sorted({
        lvl for p in p_names
        for lvl in _get_levels_for_perturbation(summary_df, p)
    })
    if not levels:
        print("  Skipping heatmap (no level data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * len(levels)), 4))

    for idx, (metric, title) in enumerate([("delta_dose_pct", "Dose Score Degradation (%)"),
                                            ("delta_dvh_pct", "DVH Score Degradation (%)")]):
        data = np.full((len(p_names), len(levels)), np.nan)
        for i, p in enumerate(p_names):
            for j, level in enumerate(levels):
                condition = f"{p}/{level}"
                row = summary_df[summary_df["perturbation"] == condition]
                if not row.empty and metric in row.columns:
                    val = row[metric].iloc[0]
                    if pd.notna(val):
                        data[i, j] = val

        im = axes[idx].imshow(data, cmap='YlOrRd', aspect='auto')
        axes[idx].set_xticks(range(len(levels)))
        axes[idx].set_xticklabels(levels)
        axes[idx].set_yticks(range(len(p_names)))
        axes[idx].set_yticklabels([PERTURBATION_LABELS[p] for p in p_names])
        axes[idx].set_title(title)

        # Add text annotations
        for i in range(len(p_names)):
            for j in range(len(levels)):
                val = data[i, j]
                if not np.isnan(val):
                    color = 'white' if val > np.nanmax(data) * 0.6 else 'black'
                    axes[idx].text(j, i, f"{val:+.1f}%", ha='center', va='center',
                                   color=color, fontsize=9, fontweight='bold')

        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    plt.tight_layout()
    fig.savefig(figures_dir / "degradation_heatmap.png", bbox_inches='tight')
    fig.savefig(figures_dir / "degradation_heatmap.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved degradation_heatmap.png/pdf")


def plot_structure_radar(details: list, figures_dir: Path) -> None:
    """Radar plot showing which structures are most affected by each perturbation."""
    # Get baseline per-structure errors
    baseline = next((d for d in details if d["condition"] == "baseline"), None)
    if not baseline or "per_structure" not in baseline:
        print("  Skipping radar plot (no baseline structure data)")
        return

    structures = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid",
                   "Esophagus", "Larynx", "Mandible", "PTV56", "PTV63", "PTV70"]

    # Use the highest available level per perturbation for the radar plot
    highest_per_perturb = {}
    for d in details:
        cond = d["condition"]
        if cond == "baseline" or "/" not in cond:
            continue
        p_name, level = cond.split("/")
        if p_name not in highest_per_perturb or level > highest_per_perturb[p_name][0]:
            highest_per_perturb[p_name] = (level, d)

    conditions_to_plot = [v[1] for v in highest_per_perturb.values()]

    if not conditions_to_plot:
        print("  Skipping radar plot (no perturbed conditions)")
        return

    # Aggregate per-structure: average across metrics for each structure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    n_structures = len(structures)
    angles = np.linspace(0, 2 * np.pi, n_structures, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions_to_plot)))

    for ci, d in enumerate(conditions_to_plot):
        values = []
        per_struct = d.get("per_structure", {})
        for s in structures:
            # Average across metrics for this structure
            struct_vals = [v for k, v in per_struct.items() if k.startswith(s)]
            values.append(np.mean(struct_vals) if struct_vals else 0)

        # Normalize by baseline
        baseline_values = []
        base_struct = baseline.get("per_structure", {})
        for s in structures:
            struct_vals = [v for k, v in base_struct.items() if k.startswith(s)]
            baseline_values.append(np.mean(struct_vals) if struct_vals else 1e-6)

        # Ratio: perturbed / baseline
        ratios = [v / max(b, 1e-6) for v, b in zip(values, baseline_values)]
        ratios += ratios[:1]

        p_name, level = d["condition"].split("/")
        label = PERTURBATION_LABELS.get(p_name, p_name) + f" {level}"
        ax.plot(angles, ratios, 'o-', color=colors[ci], label=label, linewidth=1.5, markersize=4)
        ax.fill(angles, ratios, alpha=0.1, color=colors[ci])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(structures, fontsize=8)
    ax.set_title("Per-Structure Error Ratio vs Baseline (highest level)", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fig.savefig(figures_dir / "structure_radar.png", bbox_inches='tight')
    fig.savefig(figures_dir / "structure_radar.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved structure_radar.png/pdf")


def plot_ct_slices(config: dict, figures_dir: Path) -> None:
    """Show example CT slices: clean vs perturbed for one patient per perturbation."""
    from openkbp_hn_robustness.perturbations.base import load_ct_volume

    validation_dir = project_root / config["paths"]["validation_data"]
    perturbed_root = project_root / config["paths"]["data_perturbed"]

    # Use first available patient
    start = config["patients"]["start"]
    example_pid = f"pt_{start}"
    original_dir = validation_dir / example_pid

    if not original_dir.exists():
        print(f"  Skipping CT slice plots (patient {example_pid} not found)")
        return

    original_vol, _ = load_ct_volume(original_dir)
    mid_slice = original_vol.shape[0] // 2

    p_names = list(PERTURBATION_LABELS.keys())
    n_perturb = len(p_names)

    fig, axes = plt.subplots(2, n_perturb + 1, figsize=(3 * (n_perturb + 1), 8))

    # Rotate slices 90° CCW so anatomy is upright (head at top)
    def orient(slc):
        return np.rot90(slc, k=1)

    # Original
    for row in range(2):
        if row == 0:
            axes[row, 0].imshow(orient(original_vol[mid_slice]), cmap='gray',
                                vmin=0, vmax=2000, aspect='equal')
        else:
            axes[row, 0].imshow(orient(np.zeros_like(original_vol[mid_slice])),
                                cmap='RdBu_r', vmin=-200, vmax=200, aspect='equal')
        axes[row, 0].set_title("Original" if row == 0 else "Diff")
        axes[row, 0].axis('off')

    # Each perturbation (L2)
    for pi, p_name in enumerate(p_names):
        pert_dir = perturbed_root / p_name / "L2" / example_pid
        col = pi + 1

        if pert_dir.exists():
            pert_vol, _ = load_ct_volume(pert_dir)
            axes[0, col].imshow(orient(pert_vol[mid_slice]), cmap='gray',
                                vmin=0, vmax=2000, aspect='equal')
            diff = pert_vol[mid_slice] - original_vol[mid_slice]
            axes[1, col].imshow(orient(diff), cmap='RdBu_r',
                                vmin=-200, vmax=200, aspect='equal')
        else:
            axes[0, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              transform=axes[1, col].transAxes)

        label = PERTURBATION_LABELS.get(p_name, p_name)
        axes[0, col].set_title(f"{label} L2")
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("CT (HU)")
    axes[1, 0].set_ylabel("Difference")

    plt.suptitle(f"Example CT Slices — {example_pid} (axial slice {mid_slice})", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "ct_slices.png", bbox_inches='tight')
    fig.savefig(figures_dir / "ct_slices.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved ct_slices.png/pdf")


def plot_dose_difference_maps(config: dict, figures_dir: Path) -> None:
    """Show dose difference maps: baseline prediction vs perturbed prediction."""
    from openkbp_hn_robustness.perturbations.base import VOLUME_SHAPE

    predictions_root = project_root / config["paths"]["predictions"]
    baseline_dir = predictions_root / "baseline"

    start = config["patients"]["start"]
    example_pid = f"pt_{start}"

    baseline_path = baseline_dir / f"{example_pid}.csv"
    if not baseline_path.exists():
        print(f"  Skipping dose difference maps (baseline prediction for {example_pid} not found)")
        return

    # Load baseline prediction
    df = pd.read_csv(baseline_path, index_col=0)
    baseline_dose = np.zeros(np.prod(VOLUME_SHAPE))
    baseline_dose[df.index.values] = df["data"].values
    baseline_dose = baseline_dose.reshape(VOLUME_SHAPE)

    mid_slice = VOLUME_SHAPE[0] // 2

    p_names = list(PERTURBATION_LABELS.keys())
    n_perturb = len(p_names)

    fig, axes = plt.subplots(2, n_perturb + 1, figsize=(3 * (n_perturb + 1), 8))

    # Rotate slices 90° CCW so anatomy is upright (head at top)
    def orient(slc):
        return np.rot90(slc, k=1)

    # Baseline dose
    dose_max = np.percentile(baseline_dose[baseline_dose > 0], 99) if baseline_dose.max() > 0 else 70
    axes[0, 0].imshow(orient(baseline_dose[mid_slice]), cmap='jet', vmin=0,
                       vmax=dose_max, aspect='equal')
    axes[0, 0].set_title("Baseline")
    axes[0, 0].axis('off')
    axes[1, 0].imshow(orient(np.zeros_like(baseline_dose[mid_slice])), cmap='RdBu_r',
                       vmin=-5, vmax=5, aspect='equal')
    axes[1, 0].set_title("Diff")
    axes[1, 0].axis('off')

    for pi, p_name in enumerate(p_names):
        col = pi + 1
        pert_path = predictions_root / p_name / "L2" / f"{example_pid}.csv"

        if pert_path.exists():
            df_p = pd.read_csv(pert_path, index_col=0)
            pert_dose = np.zeros(np.prod(VOLUME_SHAPE))
            pert_dose[df_p.index.values] = df_p["data"].values
            pert_dose = pert_dose.reshape(VOLUME_SHAPE)

            axes[0, col].imshow(orient(pert_dose[mid_slice]), cmap='jet', vmin=0,
                                 vmax=dose_max, aspect='equal')
            diff = pert_dose[mid_slice] - baseline_dose[mid_slice]
            axes[1, col].imshow(orient(diff), cmap='RdBu_r', vmin=-5, vmax=5,
                                 aspect='equal')
        else:
            axes[0, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              transform=axes[1, col].transAxes)

        label = PERTURBATION_LABELS.get(p_name, p_name)
        axes[0, col].set_title(f"{label} L2")
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Dose (Gy)")
    axes[1, 0].set_ylabel("Dose Diff (Gy)")

    plt.suptitle(f"Dose Predictions — {example_pid} (axial slice {mid_slice})", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "dose_difference_maps.png", bbox_inches='tight')
    fig.savefig(figures_dir / "dose_difference_maps.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved dose_difference_maps.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Visualize robustness results")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--skip-ct", action="store_true", help="Skip CT slice plots")
    parser.add_argument("--skip-dose", action="store_true", help="Skip dose difference maps")
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir / "configs" / "default.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    metrics_dir = project_root / config["paths"]["metrics"]
    figures_dir = project_root / config["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Check for results
    summary_path = metrics_dir / "summary.csv"
    if not summary_path.exists():
        print(f"No summary.csv found at {summary_path}")
        print("Run evaluate_metrics.py first!")
        return

    summary_df, details = load_results(metrics_dir)

    print(f"Generating figures in {figures_dir}")

    # 1. Degradation curves (gradual vs sudden take-off)
    print("\n1. Degradation curves")
    plot_degradation_curves(summary_df, figures_dir)

    # 2. Summary bars
    print("\n2. Summary bar chart")
    plot_summary_bars(summary_df, figures_dir)

    # 3. Degradation heatmap
    print("\n3. Degradation heatmap")
    if "delta_dose_pct" in summary_df.columns:
        plot_degradation_heatmap(summary_df, figures_dir)
    else:
        print("  Skipping (no delta columns — need baseline + perturbed results)")

    # 4. Structure radar
    print("\n4. Per-structure radar plot")
    plot_structure_radar(details, figures_dir)

    # 5. CT slices
    if not args.skip_ct:
        print("\n5. Example CT slices")
        plot_ct_slices(config, figures_dir)

    # 6. Dose difference maps
    if not args.skip_dose:
        print("\n6. Dose difference maps")
        plot_dose_difference_maps(config, figures_dir)

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
