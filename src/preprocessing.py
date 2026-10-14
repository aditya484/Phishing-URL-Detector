"""
preprocessing.py — Clean, validate, and split the dataset.

Key design decisions
--------------------
1.  Features are extracted from the raw ``URL`` column using the shared
    ``extract_url_features`` function — the same function used at inference.
2.  Duplicate URLs are removed **before** splitting so that the same URL
    never appears in both training and testing sets.
3.  The split is stratified (preserves class proportions) with a fixed seed.
4.  The processed train/test CSVs are saved to ``data/processed/`` so that
    ``evaluate.py`` can reload them without re-extracting features.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Ensure project root is on sys.path ──────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.feature_extraction import (
    extract_features_batch,
    FEATURE_NAMES,
)

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")


# ── Cleaning ────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the raw dataset.

    Steps
    -----
    1. Keep only ``URL`` and ``label`` columns (the rest are dataset-specific
       precomputed features that we intentionally do NOT use).
    2. Drop rows with missing URL or label.
    3. Ensure labels are 0 or 1.
    4. Check for conflicting labels (same URL, different label) and report.
    5. Remove exact duplicate (URL, label) pairs.
    """
    print("[CLEAN] Starting data cleaning ...")

    # 1. Keep only what we need
    if "URL" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'URL' and 'label' columns.")
    df = df[["URL", "label"]].copy()

    n_start = len(df)

    # 2. Drop missing
    df.dropna(subset=["URL", "label"], inplace=True)
    df = df[df["URL"].astype(str).str.strip().ne("")]
    n_after_missing = len(df)
    print(f"  Dropped {n_start - n_after_missing:,} rows with missing URL/label.")

    # 3. Validate labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    n_after_label = len(df)
    if n_after_label < n_after_missing:
        print(f"  Dropped {n_after_missing - n_after_label:,} rows with invalid labels.")

    # 4. Conflicting labels
    url_lower = df["URL"].str.strip().str.lower()
    label_per_url = df.groupby(url_lower)["label"].nunique()
    n_conflict = (label_per_url > 1).sum()
    if n_conflict > 0:
        print(f"  ⚠  {n_conflict:,} URLs have conflicting labels — "
              "keeping the first occurrence.")
        df = df.drop_duplicates(subset=["URL"], keep="first")
    else:
        print("  No conflicting labels found.")

    # 5. Remove exact duplicates
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["URL", "label"], keep="first")
    n_removed = n_before_dedup - len(df)
    print(f"  Removed {n_removed:,} exact-duplicate rows.")

    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] Final row count: {len(df):,}")
    return df


# ── Benign Path Augmentation ───────────────────────────────────────────────
COMMON_BENIGN_PATHS = [
    "/about",
    "/about-us",
    "/contact",
    "/contact-us",
    "/privacy",
    "/privacy-policy",
    "/terms",
    "/terms-of-service",
    "/faq",
    "/help",
    "/support",
    "/blog",
    "/blog/news",
    "/news",
    "/articles",
    "/problemset",
    "/problemset/all",
    "/trending",
    "/explore",
    "/features",
    "/pricing",
    "/products",
    "/product/details",
    "/services",
    "/docs",
    "/docs/getting-started",
    "/community",
    "/forum",
    "/login",
    "/register",
    "/signup",
    "/search",
    "/search?q=test",
    "/category/technology",
    "/app/dashboard",
]


def _augment_legitimate_urls(df: pd.DataFrame, augment_fraction: float = 0.35, random_state: int = 42) -> pd.DataFrame:
    """
    In the PhiUSIIL raw dataset, legitimate URLs were collected almost exclusively
    as root domains (e.g., https://example.com) without paths, while phishing URLs
    from feeds contain deep subpaths (e.g., /login/verify.php).
    
    This creates an artificial shortcut where any path/slash immediately triggers
    a 100% Phishing classification for completely benign URLs like LeetCode or GitHub.
    
    This function appends realistic subpaths to a subset of legitimate URLs during
    training so the model learns that having a path or query string is normal.
    """
    np.random.seed(random_state)
    df = df.copy()
    legit_mask = (df["label"] == 1)
    legit_indices = df[legit_mask].index.to_numpy()
    
    n_to_augment = int(len(legit_indices) * augment_fraction)
    chosen_indices = np.random.choice(legit_indices, size=n_to_augment, replace=False)
    
    augmented_rows = []
    for idx in chosen_indices:
        orig_url = str(df.at[idx, "URL"]).rstrip("/")
        path = np.random.choice(COMMON_BENIGN_PATHS)
        # 50% chance of adding an extension or deeper subpath
        if np.random.rand() > 0.6:
            path += "/"
        augmented_url = orig_url + path
        augmented_rows.append({"URL": augmented_url, "label": 1})
        
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        print(f"[AUGMENT] Added {len(aug_df):,} realistic path-augmented legitimate URLs to prevent path bias.")
        df = pd.concat([df, aug_df], ignore_index=True)
        
    return df


# ── Full preparation pipeline ───────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    save: bool = True,
    augment_paths: bool = True,
):
    """
    Clean → augment benign paths → extract features → split → (optionally) save processed CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (must contain ``URL`` and ``label``).
    test_size : float
        Fraction for the test set.
    random_state : int
        Seed for reproducibility.
    save : bool
        Whether to write train/test CSVs to ``data/processed/``.
    augment_paths : bool
        Whether to augment legitimate URLs with realistic benign paths.

    Returns
    -------
    X_train, X_test : pd.DataFrame  (columns = FEATURE_NAMES)
    y_train, y_test : pd.Series     (0 / 1)
    """
    df = clean_data(df)

    if augment_paths:
        df = _augment_legitimate_urls(df, augment_fraction=0.35, random_state=random_state)

    # Extract features from raw URLs
    print("\n[FEATURES] Extracting URL features ...")
    X = extract_features_batch(df["URL"])
    y = df["label"].reset_index(drop=True)

    # Sanity check for invalid feature values
    if X.isnull().any().any() or np.isinf(X.values).any():
        print("  ⚠  Found NaN/Inf in features — replacing with 0.")
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\n[SPLIT] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Train phishing ratio: {(y_train == 0).mean():.3f}")
    print(f"  Test  phishing ratio: {(y_test == 0).mean():.3f}")

    # Save processed data so evaluate.py can reload without re-extraction
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        train_df = X_train.copy()
        train_df["label"] = y_train.values
        train_df.to_csv(TRAIN_CSV, index=False)

        test_df = X_test.copy()
        test_df["label"] = y_test.values
        test_df.to_csv(TEST_CSV, index=False)
        print(f"[SAVE] Processed data → {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test


def load_processed_test():
    """Load the saved test set (features + labels)."""
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Test set not found at {TEST_CSV}.\n"
            "Run training first:  python -m src.train"
        )
    df = pd.read_csv(TEST_CSV)
    X_test = df[FEATURE_NAMES]
    y_test = df["label"]
    return X_test, y_test


# ── Run standalone: python -m src.preprocessing ─────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)
    print("\nPreprocessing finished.  Check data/processed/ for output files.")
