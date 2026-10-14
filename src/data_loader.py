"""
data_loader.py — Download and cache the PhiUSIIL Phishing URL Dataset.

The dataset is fetched from the UCI Machine Learning Repository using the
``ucimlrepo`` package and saved locally so subsequent runs are instant.

Dataset ID : 967
Rows       : 235 795
Target     : ``label`` (0 = Phishing, 1 = Legitimate)
"""

import os
import sys
import pandas as pd

# ── Ensure project root is on sys.path ──────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "phiusiil_dataset.csv")


def load_dataset(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Load the PhiUSIIL dataset.

    * If the CSV already exists on disk, read it.
    * Otherwise, download via ``ucimlrepo`` and cache it.

    Returns
    -------
    pd.DataFrame
        Full dataset with at least ``URL`` and ``label`` columns.
    """
    if os.path.exists(csv_path):
        print(f"[DATA] Loading cached dataset from {csv_path}")
        return pd.read_csv(csv_path, low_memory=False)

    print("[DATA] Downloading dataset from UCI ML Repository (id=967) ...")
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=967)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[DATA] Dataset saved → {csv_path}  ({len(df):,} rows)")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Rows           : {df.shape[0]:,}")
    print(f"  Columns        : {df.shape[1]}")
    print(f"  Memory (MB)    : {df.memory_usage(deep=True).sum() / 1e6:.1f}")

    # Missing values
    n_missing = df.isnull().sum().sum()
    print(f"  Missing values : {n_missing:,}")

    # Duplicates (by URL column, if present)
    if "URL" in df.columns:
        n_dup = df["URL"].duplicated().sum()
        print(f"  Duplicate URLs : {n_dup:,}")

    # Class balance
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        print(f"\n  Class distribution:")
        print(f"    Phishing   (0) : {counts.get(0, 0):>8,}  "
              f"({counts.get(0, 0) / total * 100:.1f}%)")
        print(f"    Legitimate (1) : {counts.get(1, 0):>8,}  "
              f"({counts.get(1, 0) / total * 100:.1f}%)")
    print("=" * 55)


# ── Run standalone: python -m src.data_loader ────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    inspect_dataset(df)
    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>2}. {col}  ({df[col].dtype})")
