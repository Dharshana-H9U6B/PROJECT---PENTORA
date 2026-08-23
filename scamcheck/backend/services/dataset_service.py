"""
Dataset service — loads, validates, and normalizes datasets for ML training.

Supports configurable column mappings and label normalization.
Developers only need to update config.yaml to use a new dataset.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.config import get_config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Configurable dataset loader for scam classification datasets.

    Usage:
        loader = DatasetLoader(config)
        df = loader.load()
        df = loader.normalize(df)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.dataset_cfg = self.config.get("dataset", {})

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from the configured path.

        Returns:
            Raw DataFrame.

        Raises:
            FileNotFoundError: If dataset file does not exist.
            ValueError: If required columns are missing.
        """
        root = Path(__file__).parent.parent.parent
        path = root / self.dataset_cfg.get("path", "data/raw/scam_dataset.csv")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {path}\n"
                "Place a CSV file there or update 'dataset.path' in config.yaml."
            )

        logger.info(f"Loading dataset from: {path}")
        df = pd.read_csv(str(path))
        logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        return df

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that the required columns exist in the DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        text_col = self.dataset_cfg.get("text_column", "message")
        label_col = self.dataset_cfg.get("label_column", "label")

        missing = []
        if text_col not in df.columns:
            missing.append(f"text_column '{text_col}'")
        if label_col not in df.columns:
            missing.append(f"label_column '{label_col}'")

        if missing:
            raise ValueError(
                f"Missing columns in dataset: {', '.join(missing)}.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Update dataset.text_column / dataset.label_column in config.yaml."
            )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the dataset with standard column names and binary labels.

        Returns:
            DataFrame with columns: ['text', 'label'] where label is 0 or 1.
        """
        self.validate_columns(df)

        text_col = self.dataset_cfg.get("text_column", "message")
        label_col = self.dataset_cfg.get("label_column", "label")

        # Rename to standard names
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]

        # Drop rows with missing values
        before = len(df)
        df = df.dropna(subset=["text", "label"])
        after = len(df)
        if before != after:
            logger.info(f"Dropped {before - after} rows with missing values.")

        # Normalize labels to 0/1
        df["label"] = df["label"].apply(self._normalize_label)

        # Drop rows where label normalization failed
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        logger.info(f"Normalized dataset: {len(df)} rows.")
        return df

    def _normalize_label(self, raw_label) -> Optional[int]:
        """
        Map a raw label value to 0 (legitimate) or 1 (scam).
        Reads label mapping from config.
        """
        label_config = self.dataset_cfg.get("labels", {})
        scam_labels = set(str(v).lower() for v in label_config.get("scam", ["scam", "spam", "fraud", "1"]))
        legit_labels = set(str(v).lower() for v in label_config.get("legitimate", ["legitimate", "ham", "safe", "0"]))

        raw_str = str(raw_label).lower().strip()
        if raw_str in scam_labels:
            return 1
        if raw_str in legit_labels:
            return 0

        logger.warning(f"Unknown label value: '{raw_label}'. Dropping row.")
        return None


def clean_text_for_ml(text: str) -> str:
    """
    Clean text for ML training (more aggressive than analysis cleaning).
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs (ML model doesn't need them verbatim)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)

    # Remove phone numbers
    text = re.sub(r"[\+\d][\d\s\-\(\)]{8,}", " PHONE ", text)

    # Remove special characters but keep letters, digits, spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
