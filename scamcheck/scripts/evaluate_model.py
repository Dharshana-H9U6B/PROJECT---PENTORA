#!/usr/bin/env python3
"""
ScamCheck Model Evaluation Script

Evaluates the saved model artifacts on a test set.

Usage:
    python scripts/evaluate_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import joblib

from backend.config import get_config
from backend.services.dataset_service import DatasetLoader, clean_text_for_ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)


def evaluate():
    print("=" * 60)
    print("ScamCheck — Model Evaluation")
    print("=" * 60)

    config = get_config()
    model_cfg = config.get("model", {})
    model_dir = Path(__file__).parent.parent / "models"

    classifier_path = model_dir / model_cfg.get("classifier_file", "scam_classifier.joblib")
    vectorizer_path = model_dir / model_cfg.get("vectorizer_file", "tfidf_vectorizer.joblib")

    if not classifier_path.exists() or not vectorizer_path.exists():
        print("[ERROR] Model artifacts not found. Run train_model.py first.")
        sys.exit(1)

    clf = joblib.load(str(classifier_path))
    vectorizer = joblib.load(str(vectorizer_path))

    loader = DatasetLoader(config)
    df = loader.load()
    df = loader.normalize(df)
    df["clean_text"] = df["text"].apply(clean_text_for_ml)

    X = df["clean_text"].values
    y = df["label"].values

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=float(model_cfg.get("test_size", 0.2)),
        random_state=int(model_cfg.get("random_state", 42)),
        stratify=y
    )

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

    print(f"\nTest samples: {len(X_test)}")
    print(f"Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:    {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:       {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:     {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:      {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"], zero_division=0))
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
