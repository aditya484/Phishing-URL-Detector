"""
train.py — Train Random Forest Classifier and baseline model on extracted URL features.

Workflow:
1. Load dataset via data_loader.py
2. Clean, extract features, and split via preprocessing.py
3. Fit baseline model (Logistic Regression) for comparison
4. Fit Random Forest Classifier wrapped in a Scikit-learn Pipeline
5. Evaluate on held-out test data — Phishing (class 0) is the positive class
6. Save model package + metadata to models/phishing_model.joblib
7. Save confusion matrix image to reports/figures/confusion_matrix.png
8. Save metrics dict to reports/metrics.json

Run standalone:
    python -m src.train
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import load_dataset
from src.preprocessing import prepare_data
from src.feature_extraction import FEATURE_NAMES, CLASS_NAMES

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR          = os.path.join(_PROJECT_ROOT, "models")
REPORTS_DIR         = os.path.join(_PROJECT_ROOT, "reports")
FIGURES_DIR         = os.path.join(REPORTS_DIR, "figures")
MODEL_FILE          = os.path.join(MODELS_DIR, "phishing_model.joblib")
METRICS_FILE        = os.path.join(REPORTS_DIR, "metrics.json")
CONFUSION_MATRIX_IMG = os.path.join(FIGURES_DIR, "confusion_matrix.png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _phishing_proba_column(pipeline, X):
    """
    Return the column of predict_proba that corresponds to class 0 (Phishing).

    sklearn's predict_proba columns follow the order in clf.classes_, which is
    sorted.  For a balanced dataset [0, 1] column 0 is always Phishing — but
    we look it up explicitly to be safe.
    """
    clf = pipeline.named_steps["classifier"]
    classes = list(clf.classes_)
    phish_col = classes.index(0)          # index of class-0 in probability matrix
    return pipeline.predict_proba(X)[:, phish_col]


def compute_metrics(y_true, y_pred, y_proba_phishing):
    """
    Compute a metrics dict with Phishing (class 0) as the primary positive class.

    Parameters
    ----------
    y_true            : array-like  — true labels (0 / 1)
    y_pred            : array-like  — predicted labels
    y_proba_phishing  : array-like  — probability of class 0 (Phishing)
    """
    p_phish  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r_phish  = recall_score   (y_true, y_pred, pos_label=0, zero_division=0)
    f1_phish = f1_score       (y_true, y_pred, pos_label=0, zero_division=0)

    p_legit  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_legit  = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
    f1_legit = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)

    acc = accuracy_score(y_true, y_pred)

    # ROC-AUC: binary perspective where Phishing = positive (1), Legit = 0
    y_binary = (np.array(y_true) == 0).astype(int)
    auc = roc_auc_score(y_binary, y_proba_phishing)

    return {
        "accuracy": round(float(acc), 6),
        "roc_auc":  round(float(auc), 6),
        "phishing_metrics": {
            "precision": round(float(p_phish), 6),
            "recall":    round(float(r_phish), 6),
            "f1_score":  round(float(f1_phish), 6),
        },
        "legitimate_metrics": {
            "precision": round(float(p_legit), 6),
            "recall":    round(float(r_legit), 6),
            "f1_score":  round(float(f1_legit), 6),
        },
    }


# ── Main training function ────────────────────────────────────────────────────

def train_pipeline(max_samples=None):
    """Execute the complete training and model-saving workflow."""

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHISHING WEBSITE DETECTOR — TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load dataset
    raw_df = load_dataset()
    if max_samples and len(raw_df) > max_samples:
        print(f"[INFO] Subsampling to {max_samples:,} rows.")
        raw_df = raw_df.sample(n=max_samples, random_state=42)

    # 2. Preprocess & extract consistent features
    X_train, X_test, y_train, y_test = prepare_data(
        raw_df, test_size=0.20, random_state=42
    )

    # 3. Baseline — Logistic Regression
    print("\n[TRAIN] Baseline: Logistic Regression ...")
    base_pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )),
    ])
    base_pipe.fit(X_train, y_train)
    base_pred  = base_pipe.predict(X_test)
    base_proba = _phishing_proba_column(base_pipe, X_test)
    base_metrics = compute_metrics(y_test, base_pred, base_proba)
    print(
        f"  Baseline  Accuracy={base_metrics['accuracy']:.4f}  "
        f"Phishing-F1={base_metrics['phishing_metrics']['f1_score']:.4f}"
    )

    # 4. Primary model — Random Forest (regularized to prevent path-depth bias)
    print("\n[TRAIN] Primary: Random Forest (200 trees, max_depth=16, regularized) ...")
    rf_pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_split=10,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    print("  Training complete.")

    # 5. Evaluate on held-out test set
    print("\n[EVALUATE] Held-out test set ...")
    y_pred    = rf_pipe.predict(X_test)
    proba_phi = _phishing_proba_column(rf_pipe, X_test)
    metrics   = compute_metrics(y_test, y_pred, proba_phi)

    print("\n" + "-" * 55)
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC (Phishing) : {metrics['roc_auc']:.4f}")
    print(f"  Phishing Precision : {metrics['phishing_metrics']['precision']:.4f}")
    print(f"  Phishing Recall    : {metrics['phishing_metrics']['recall']:.4f}")
    print(f"  Phishing F1-Score  : {metrics['phishing_metrics']['f1_score']:.4f}")
    print(f"  Legitimate F1      : {metrics['legitimate_metrics']['f1_score']:.4f}")
    print("-" * 55)
    print("\nClassification Report (0=Phishing, 1=Legitimate):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Phishing (0)", "Legitimate (1)"],
        zero_division=0,
    ))

    # 6. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted Phishing", "Predicted Legitimate"],
        yticklabels=["Actual Phishing",    "Actual Legitimate"],
    )
    ax.set_title("Confusion Matrix — Random Forest (Test Set)", fontsize=12, pad=12)
    ax.set_ylabel("Actual Label",    fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG, dpi=200)
    plt.close()
    print(f"[PLOT] Confusion matrix → {CONFUSION_MATRIX_IMG}")

    # 7. Save metrics.json
    report = {
        "dataset":       "PhiUSIIL Phishing URL Dataset (UCI ID 967)",
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "primary_model": {
            "name": "RandomForestClassifier",
            "parameters": {
                "n_estimators":    200,
                "max_depth":       25,
                "class_weight":    "balanced",
                "random_state":    42,
            },
            "metrics": metrics,
        },
        "baseline_model": {
            "name":    "LogisticRegression",
            "metrics": base_metrics,
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[METRICS] Saved → {METRICS_FILE}")

    # 8. Save model package
    package = {
        "pipeline":      rf_pipe,
        "feature_names": FEATURE_NAMES,
        "class_names":   CLASS_NAMES,
        "version":       "1.0.0",
        "description":   (
            "Random Forest trained on 30 URL-structural features extracted "
            "from the PhiUSIIL dataset. Same features used at inference."
        ),
    }
    joblib.dump(package, MODEL_FILE)
    print(f"[SAVE] Model artifact → {MODEL_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    train_pipeline()
