"""
main.py — Master Execution Pipeline for Phishing Website Detection.

Executes the complete project lifecycle:
1. Data Acquisition & Validation
2. Exploratory Data Analysis (generates figures)
3. Preprocessing & Feature Extraction
4. Model Training (Baseline & Random Forest)
5. Comprehensive Evaluation & Plot Generation

Run with:
    python main.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.eda import run_eda
from src.train import train_pipeline
from src.evaluate import evaluate_saved_model


def main():
    print("\n" + "=" * 70)
    print("  PHISHING WEBSITE DETECTION — MASTER WORKFLOW")
    print("=" * 70)

    # 1. Exploratory Data Analysis
    print("\n[PHASE 1] Executing Exploratory Data Analysis (EDA)...")
    try:
        run_eda(sample_size=25000)
    except Exception as e:
        print(f"  Note during EDA: {e}")

    # 2. Training Pipeline
    print("\n[PHASE 2] Executing Training Pipeline...")
    train_pipeline()

    # 3. Comprehensive Evaluation
    print("\n[PHASE 3] Running Independent Model Evaluation...")
    evaluate_saved_model()

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("  To launch the web interface, run:")
    print("      streamlit run app.py")
    print("  To run unit tests, run:")
    print("      python -m pytest")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
