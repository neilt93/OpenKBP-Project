#!/usr/bin/env python3
"""
Plot Adversarial Attack Results

Creates visualizations showing how model performance degrades under attack:
- Histogram of dose errors at each epsilon
- Box plots of per-patient MAE vs epsilon
- Line plot of mean MAE vs epsilon with error bands

Usage:
    python plot_adversarial.py --results-dir adversarial_results/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(results_dir: Path) -> dict:
    """Load all result files from directory."""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file) as f:
        summary = json.load(f)

    # Load detailed results for each attack/epsilon
    detailed = {}
    for attack in summary["attacks"]:
        detailed[attack] = {}
        for eps in summary["epsilons"]:
            result_file = results_dir / f"{attack}_eps{eps:.4f}.json"
            if result_file.exists():
                with open(result_file) as f:
                    detailed[attack][eps] = json.load(f)

    return summary, detailed


def plot_mae_vs_epsilon(summary: dict, output_dir: Path):
    """Plot mean MAE vs epsilon for each attack type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"fgsm": "#1f77b4", "pgd": "#ff7f0e"}
    markers = {"fgsm": "o", "pgd": "s"}

    for attack in summary["attacks"]:
        epsilons = []
        means = []
        stds = []

        for eps_str, stats in summary["results"][attack].items():
            epsilons.append(float(eps_str))
            means.append(stats["mean_mae"])
            stds.append(stats["std_mae"])

        # Sort by epsilon
        sorted_idx = np.argsort(epsilons)
        epsilons = np.array(epsilons)[sorted_idx]
        means = np.array(means)[sorted_idx]
        stds = np.array(stds)[sorted_idx]

        # Plot with error bands
        ax.plot(
            epsilons, means,
            color=colors.get(attack, "gray"),
            marker=markers.get(attack, "o"),
            linewidth=2,
            markersize=8,
            label=attack.upper(),
        )
        ax.fill_between(
            epsilons,
            means - stds,
            means + stds,
            color=colors.get(attack, "gray"),
            alpha=0.2,
        )

    ax.set_xlabel("Epsilon (perturbation magnitude)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (Gy)", fontsize=12)
    ax.set_title("Dose Prediction Error vs Adversarial Perturbation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add baseline reference
    baseline_mae = summary["results"][summary["attacks"][0]]["0.0"]["mean_mae"]
    ax.axhline(y=baseline_mae, color="gray", linestyle="--", alpha=0.5)
    ax.text(
        ax.get_xlim()[1] * 0.7, baseline_mae * 1.02,
        f"Baseline: {baseline_mae:.2f} Gy",
        fontsize=10, color="gray"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "mae_vs_epsilon.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'mae_vs_epsilon.png'}")


def plot_patient_boxplots(detailed: dict, output_dir: Path):
    """Create box plots showing per-patient MAE distribution at each epsilon."""
    for attack, eps_results in detailed.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        epsilons = sorted(eps_results.keys())
        data = []
        labels = []

        for eps in epsilons:
            if eps in eps_results:
                patient_maes = [p["mae"] for p in eps_results[eps]["patients"]]
                data.append(patient_maes)
                labels.append(f"{eps:.3f}")

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color gradient based on epsilon
        cmap = plt.cm.Reds
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(0.3 + 0.7 * i / len(data)))
            patch.set_alpha(0.7)

        ax.set_xlabel("Epsilon", fontsize=12)
        ax.set_ylabel("MAE per Patient (Gy)", fontsize=12)
        ax.set_title(f"{attack.upper()} Attack: Per-Patient Error Distribution", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{attack}.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / f'boxplot_{attack}.png'}")


def plot_error_histograms(detailed: dict, output_dir: Path):
    """Plot histograms of voxel-wise errors at selected epsilons."""
    # Select a few representative epsilons
    target_eps = [0.0, 0.01, 0.05, 0.1]

    for attack, eps_results in detailed.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(target_eps)))

        for i, (eps, ax) in enumerate(zip(target_eps, axes)):
            if eps not in eps_results:
                # Find closest epsilon
                available = list(eps_results.keys())
                eps = min(available, key=lambda x: abs(x - eps))

            result = eps_results[eps]

            # Get patient MAEs for histogram
            patient_maes = [p["mae"] for p in result["patients"]]

            ax.hist(
                patient_maes,
                bins=15,
                color=colors[i],
                edgecolor="black",
                alpha=0.7,
            )

            mean_mae = result["summary"]["mean_mae"]
            ax.axvline(mean_mae, color="red", linestyle="--", linewidth=2)

            ax.set_xlabel("MAE (Gy)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"ε = {eps:.3f} (mean = {mean_mae:.2f} Gy)", fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{attack.upper()} Attack: Patient MAE Distributions",
            fontsize=14, y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_dir / f"histogram_{attack}.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / f'histogram_{attack}.png'}")


def plot_comparison(summary: dict, output_dir: Path):
    """Plot FGSM vs PGD comparison if both available."""
    if len(summary["attacks"]) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data for both attacks
    fgsm_data = summary["results"].get("fgsm", {})
    pgd_data = summary["results"].get("pgd", {})

    # Common epsilons
    fgsm_eps = set(fgsm_data.keys())
    pgd_eps = set(pgd_data.keys())
    common_eps = sorted([float(e) for e in fgsm_eps & pgd_eps])

    if not common_eps:
        return

    fgsm_mae = [fgsm_data[str(e)]["mean_mae"] for e in common_eps]
    pgd_mae = [pgd_data[str(e)]["mean_mae"] for e in common_eps]

    x = np.arange(len(common_eps))
    width = 0.35

    bars1 = ax.bar(x - width/2, fgsm_mae, width, label="FGSM", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + width/2, pgd_mae, width, label="PGD", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Mean MAE (Gy)", fontsize=12)
    ax.set_title("FGSM vs PGD Attack Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.3f}" for e in common_eps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "fgsm_vs_pgd.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'fgsm_vs_pgd.png'}")


def generate_report(summary: dict, output_dir: Path):
    """Generate a text report summarizing results."""
    report = []
    report.append("=" * 60)
    report.append("ADVERSARIAL ATTACK EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"\nModel: {summary['model']}")
    report.append(f"Timestamp: {summary['timestamp']}")
    report.append(f"Attacks: {', '.join(summary['attacks'])}")
    report.append(f"Epsilons: {summary['epsilons']}")

    for attack in summary["attacks"]:
        report.append(f"\n{'-' * 40}")
        report.append(f"{attack.upper()} ATTACK RESULTS")
        report.append(f"{'-' * 40}")

        eps_results = summary["results"][attack]
        baseline = eps_results.get("0.0", eps_results.get("0", {}))
        baseline_mae = baseline.get("mean_mae", 0)

        report.append(f"\n{'Epsilon':<10} {'Mean MAE':<12} {'Std MAE':<12} {'Degradation':<12}")
        report.append("-" * 50)

        for eps_str in sorted(eps_results.keys(), key=float):
            stats = eps_results[eps_str]
            eps = float(eps_str)
            degradation = (stats["mean_mae"] / baseline_mae - 1) * 100 if baseline_mae > 0 else 0
            report.append(
                f"{eps:<10.4f} {stats['mean_mae']:<12.3f} {stats['std_mae']:<12.3f} {degradation:>+10.1f}%"
            )

    report_text = "\n".join(report)

    report_file = output_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\nSaved: {report_file}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Plot adversarial attack results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing adversarial evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results-dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    summary, detailed = load_results(results_dir)

    print("\nGenerating plots...")
    plot_mae_vs_epsilon(summary, output_dir)
    plot_patient_boxplots(detailed, output_dir)
    plot_error_histograms(detailed, output_dir)
    plot_comparison(summary, output_dir)
    generate_report(summary, output_dir)

    print(f"\nAll plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
