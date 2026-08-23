#!/usr/bin/env python3
"""
ScamCheck ML Model Training Script

Usage:
    python scripts/train_model.py

This script:
    1. Loads the configured dataset
    2. Validates and normalizes columns and labels
    3. Cleans text
    4. Splits into train/validation/test sets
    5. Trains TF-IDF vectorizer
    6. Trains Logistic Regression classifier
    7. Evaluates and prints metrics
    8. Saves model artifacts to models/

Dataset configuration is read from config.yaml.
To use a different dataset, update config.yaml — no code changes required.
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

from backend.config import get_config
from backend.services.dataset_service import DatasetLoader, clean_text_for_ml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train():
    print("=" * 60)
    print("ScamCheck — ML Model Training")
    print("=" * 60)
    print()

    config = get_config()
    model_cfg = config.get("model", {})

    # ── 1. Load dataset ──────────────────────────────────────────
    print("Loading dataset...")
    loader = DatasetLoader(config)
    try:
        df = loader.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── 2. Validate and normalize ─────────────────────────────────
    try:
        df = loader.normalize(df)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"Dataset size:          {len(df)} samples")
    print(f"Class distribution:")
    dist = df["label"].value_counts()
    scam_count = dist.get(1, 0)
    legit_count = dist.get(0, 0)
    print(f"  Scam (1):            {scam_count} ({scam_count/len(df)*100:.1f}%)")
    print(f"  Legitimate (0):      {legit_count} ({legit_count/len(df)*100:.1f}%)")
    print()

    # ── 3. Clean text ─────────────────────────────────────────────
    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text_for_ml)

    # ── 4. Split ──────────────────────────────────────────────────
    test_size = float(model_cfg.get("test_size", 0.2))
    val_size = float(model_cfg.get("validation_size", 0.1))
    random_state = int(model_cfg.get("random_state", 42))

    X = df["clean_text"].values
    y = df["label"].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    print(f"Train samples:         {len(X_train)}")
    print(f"Validation samples:    {len(X_val)}")
    print(f"Test samples:          {len(X_test)}")
    print()

    # ── 5. TF-IDF ────────────────────────────────────────────────
    print("Training TF-IDF vectorizer...")
    max_features = int(model_cfg.get("tfidf_max_features", 10000))
    ngram_range = tuple(model_cfg.get("tfidf_ngram_range", [1, 2]))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # ── 6. Train classifier ───────────────────────────────────────
    print("Training Logistic Regression classifier...")
    base_clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        class_weight="balanced",
    )
    base_clf.fit(X_train_tfidf, y_train)

    # ── 7. Validation ─────────────────────────────────────────────
    y_val_pred = base_clf.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy:   {val_acc:.4f}")

    # ── 8. Final evaluation on test set ──────────────────────────
    print()
    print("Final Evaluation (Test Set)")
    print("-" * 40)
    y_test_pred = base_clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                 target_names=["Legitimate", "Scam"],
                                 zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()

    print(
        "\nNOTE: These metrics are based on the provided dataset and do not represent\n"
        "      real-world scam detection performance. Expand the dataset for better results.\n"
    )

    # ── 9. Save artifacts ─────────────────────────────────────────
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    classifier_path = model_dir / model_cfg.get("classifier_file", "scam_classifier.joblib")
    vectorizer_path = model_dir / model_cfg.get("vectorizer_file", "tfidf_vectorizer.joblib")

    joblib.dump(base_clf, str(classifier_path))
    joblib.dump(vectorizer, str(vectorizer_path))

    print(f"Model saved:      {classifier_path}")
    print(f"Vectorizer saved: {vectorizer_path}")
    print()
    print("Model saved successfully.")
    print("=" * 60)


if __name__ == "__main__":
    train()
