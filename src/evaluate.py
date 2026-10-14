"""
evaluate.py — Standalone Model Evaluation & Reporting.

Loads the saved model (models/phishing_model.joblib) and evaluates it on the
held-out test set saved in data/processed/test.csv.

Generates:
  - Full classification report with Phishing (0) as positive class
  - ROC Curve plot  →  reports/figures/roc_curve.png
  - Feature Importance plot  →  reports/figures/feature_importances.png
  - Plain-language security impact analysis (false negatives vs false positives)

Run standalone:
    python -m src.evaluate
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing import load_processed_test
from src.feature_extraction import FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(_PROJECT_ROOT, "models", "phishing_model.joblib")
FIGURES_DIR      = os.path.join(_PROJECT_ROOT, "reports", "figures")
ROC_PLOT_PATH    = os.path.join(FIGURES_DIR, "roc_curve.png")
FEAT_IMP_PATH    = os.path.join(FIGURES_DIR, "feature_importances.png")


def _phishing_proba_column(pipeline, X):
    """Return phishing (class-0) probability column, looked up from clf.classes_."""
    clf = pipeline.named_steps["classifier"]
    classes = list(clf.classes_)
    phish_col = classes.index(0)
    return pipeline.predict_proba(X)[:, phish_col]


def evaluate_saved_model():
    """Evaluate the saved model on the held-out test set."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHISHING WEBSITE DETECTOR — MODEL EVALUATION")
    print("=" * 65)

    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at:\n  {MODEL_PATH}")
        print("Run training first:  python -m src.train")
        sys.exit(1)

    print(f"\n[1] Loading model from:\n    {MODEL_PATH}")
    package  = joblib.load(MODEL_PATH)
    pipeline = package["pipeline"]
    feature_names = package.get("feature_names", FEATURE_NAMES)
    print(f"    Model version : {package.get('version', 'unknown')}")

    # 2. Load test data
    print("\n[2] Loading held-out test data ...")
    X_test, y_test = load_processed_test()
    print(f"    {len(X_test):,} samples  |  "
          f"Phishing: {(y_test == 0).sum():,}  Legitimate: {(y_test == 1).sum():,}")

    # 3. Predictions
    print("\n[3] Generating predictions ...")
    y_pred      = pipeline.predict(X_test)
    phish_proba = _phishing_proba_column(pipeline, X_test)

    # 4. Metrics
    acc      = accuracy_score(y_test, y_pred)
    prec_p   = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_p    = recall_score   (y_test, y_pred, pos_label=0, zero_division=0)
    f1_p     = f1_score       (y_test, y_pred, pos_label=0, zero_division=0)

    y_binary = (np.array(y_test) == 0).astype(int)
    auc      = roc_auc_score(y_binary, phish_proba)

    print("\n" + "=" * 52)
    print("  SECURITY METRICS  (Positive Class = Phishing)")
    print("=" * 52)
    print(f"  Accuracy           : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Phishing Precision : {prec_p:.4f}")
    print(f"  Phishing Recall    : {rec_p:.4f}")
    print(f"  Phishing F1-Score  : {f1_p:.4f}")
    print(f"  ROC-AUC (Phishing) : {auc:.4f}")
    print("=" * 52)

    print("\nFull Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Phishing (0)", "Legitimate (1)"],
        zero_division=0,
    ))

    # 5. Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fn_phish = cm[0, 1]   # Phishing predicted as Legitimate — DANGEROUS
    fp_phish = cm[1, 0]   # Legitimate predicted as Phishing — inconvenient

    print("[4] Security Impact:")
    print(f"  False Negatives (Phishing → Predicted Legitimate) : {fn_phish:,}")
    print("    ↳ High risk — user may trust a malicious site.")
    print(f"  False Positives (Legitimate → Predicted Phishing) : {fp_phish:,}")
    print("    ↳ Low risk — user sees an unnecessary warning.")

    # 6. ROC curve plot
    print("\n[5] Saving ROC Curve ...")
    fpr, tpr, _ = roc_curve(y_binary, phish_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#e74c3c", lw=2,
             label=f"Random Forest  (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
             label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate\n(Legitimate sites flagged as Phishing)", fontsize=10)
    plt.ylabel("True Positive Rate\n(Phishing sites correctly detected)", fontsize=10)
    plt.title("ROC Curve — Phishing Detection (Positive = Phishing)", fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=200)
    plt.close()
    print(f"    Saved → {ROC_PLOT_PATH}")

    # 7. Feature importance plot
    print("\n[6] Saving Feature Importance plot ...")
    clf = pipeline.named_steps.get("classifier")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices     = np.argsort(importances)[::-1][:15]  # top 15
        top_names   = [feature_names[i] for i in indices]
        top_scores  = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        # Use hue= so seaborn does not raise a FutureWarning about palette
        colors = sns.color_palette("viridis", len(top_names))
        ax.barh(top_names[::-1], top_scores[::-1], color=colors[::-1])
        ax.set_title("Top 15 URL Feature Importances (Random Forest)",
                     fontsize=12, pad=12)
        ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(FEAT_IMP_PATH, dpi=200)
        plt.close()
        print(f"    Saved → {FEAT_IMP_PATH}")

    print("\nEvaluation complete.")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_saved_model()
