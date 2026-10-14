"""
eda.py — Exploratory Data Analysis for the PhiUSIIL Phishing URL Dataset.

Generates and saves figures to reports/figures/:
  1. class_distribution.png
  2. url_length_distribution.png
  3. selected_features_distribution.png  (box plots per class)
  4. correlation_heatmap.png

Run standalone:
    python -m src.eda
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import load_dataset
from src.feature_extraction import extract_features_batch

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "reports", "figures")


def run_eda(sample_size=25_000):
    """
    Run EDA and save all figures.

    Parameters
    ----------
    sample_size : int
        Number of rows to sample for feature-extraction plots.
        The full dataset is still used for the class distribution chart.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── 1. Load raw dataset ───────────────────────────────────────────────
    df = load_dataset()
    print(f"\n[1] Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Missing values
    n_missing = df.isnull().sum().sum()
    print(f"    Missing values : {n_missing:,}")
    if "URL" in df.columns:
        print(f"    Duplicate URLs : {df['URL'].duplicated().sum():,}")

    # ── 2. Class Distribution Bar Chart ──────────────────────────────────
    print("\n[2] Class distribution chart ...")
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index()
        total  = len(df)

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            ["Phishing (0)", "Legitimate (1)"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#e74c3c", "#2ecc71"],
            width=0.5,
            edgecolor="black",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + total * 0.005,
                f"{int(h):,}\n({h / total * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_title("Class Distribution in Dataset", fontsize=14, pad=14)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        _save(fig, "class_distribution.png")

    # ── 3. Sample & extract URL features ─────────────────────────────────
    sample_n = min(sample_size, len(df))
    print(f"\n[3] Extracting URL features on a {sample_n:,}-row sample ...")
    eda_df = df.sample(n=sample_n, random_state=42).copy()
    X      = extract_features_batch(eda_df["URL"], show_progress=False)
    X["label"] = eda_df["label"].values

    # ── 4. URL-length distribution ────────────────────────────────────────
    print("\n[4] URL length distribution ...")
    p99 = float(np.percentile(X["url_length"], 99))
    sub = X[X["url_length"] <= p99].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, name in [(0, "#e74c3c", "Phishing"), (1, "#2ecc71", "Legitimate")]:
        vals = sub.loc[sub["label"] == label, "url_length"]
        ax.hist(vals, bins=60, alpha=0.55, color=color, label=name, density=True)
    ax.set_title("URL Length Distribution by Class (≤ 99th percentile)", fontsize=13, pad=12)
    ax.set_xlabel("Total URL Length (characters)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, "url_length_distribution.png")

    # ── 5. Box plots for selected features ───────────────────────────────
    print("\n[5] Selected feature distributions ...")
    selected = [
        "num_dots", "num_hyphens", "num_digits",
        "num_special_chars", "url_entropy", "path_depth",
    ]
    selected = [c for c in selected if c in X.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    palette = {0: "#e74c3c", 1: "#2ecc71"}

    for i, col in enumerate(selected):
        p98  = float(np.percentile(X[col], 98))
        sub2 = X[X[col] <= max(p98, 1)].copy()
        sub2["Class"] = sub2["label"].map({0: "Phishing", 1: "Legitimate"})

        # seaborn ≥ 0.12: pass x, y, hue to boxplot without FutureWarning
        sns.boxplot(
            data=sub2, x="Class", y=col,
            hue="Class", palette={"Phishing": "#e74c3c", "Legitimate": "#2ecc71"},
            legend=False, showfliers=False, ax=axes[i],
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Value", fontsize=9)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Key URL Feature Distributions by Class", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "selected_features_distribution.png")

    # ── 6. Correlation heatmap ────────────────────────────────────────────
    print("\n[6] Correlation heatmap ...")
    heatmap_cols = [
        "url_length", "hostname_length", "path_length", "num_dots",
        "num_hyphens", "num_digits", "num_special_chars", "digit_ratio",
        "special_char_ratio", "path_depth", "url_entropy", "is_https", "label",
    ]
    heatmap_cols = [c for c in heatmap_cols if c in X.columns]
    corr = X[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        square=True, linewidths=0.4,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Correlation Heatmap — Extracted Features vs Label",
                 fontsize=13, pad=12)
    plt.tight_layout()
    _save(fig, "correlation_heatmap.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[7] Observations:")
    print("  • Phishing URLs tend to be longer with more digits and special chars.")
    print("  • Legitimate URLs show higher `is_https` rates on average.")
    print("  • `url_entropy` captures randomness introduced by DGA-style hostnames.")
    print("  • No single feature is a definitive discriminator.")
    print("  • A Random Forest effectively combines many weak signals into a")
    print("    strong, nonlinear classifier.")
    print("=" * 60)


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")


if __name__ == "__main__":
    run_eda()
