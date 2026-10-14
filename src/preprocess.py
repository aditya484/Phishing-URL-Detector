"""
preprocess.py — Clean and prepare the dataset for training.

Steps
-----
1. Drop non-numeric / identifier columns (FILENAME, URL, Domain, TLD).
2. Split into features (X) and target (y).
3. Train / test split (80-20, stratified) — done BEFORE any fitting so that
   no information from the test set leaks into training.
4. Feature scaling with StandardScaler (fitted on train only, then applied to
   test).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns to drop: identifiers and raw text that the model cannot use directly
COLUMNS_TO_DROP = ["FILENAME", "URL", "Domain", "TLD"]

# Target column name
TARGET_COLUMN = "label"


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Preprocess the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from ``data_loader.load_dataset()``.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    scale : bool
        Whether to standardise features (recommended for some models but
        Random Forest is tree-based and does not strictly need it; we include
        it for good practice and to support other models).

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Processed splits.
    feature_names : list[str]
        Names of the features used.
    scaler : StandardScaler or None
        Fitted scaler (needed at prediction time).
    """

    # 1. Drop identifier / text columns (keep only numeric features + label)
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df_clean = df.drop(columns=cols_present)
    print(f"[PREPROCESS] Dropped columns: {cols_present}")

    # 2. Separate features and target
    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    # Ensure everything is numeric (safety net)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    feature_names = list(X.columns)
    print(f"[PREPROCESS] Using {len(feature_names)} features.")

    # 3. Train / Test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[PREPROCESS] Train size: {X_train.shape[0]:,}  |  "
        f"Test size: {X_test.shape[0]:,}"
    )

    # 4. Scaling (fit on train only → transform both)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("[PREPROCESS] Features scaled (StandardScaler).")

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    X_train, X_test, y_train, y_test, feat, scaler = preprocess(df)
    print(f"\nFeature sample: {feat[:5]} ...")
