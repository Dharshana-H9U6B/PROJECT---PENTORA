"""
Local ML provider implementation.

Uses a scikit-learn TF-IDF + classifier pipeline trained locally.
Falls back gracefully if the model files are not yet present.
"""

import logging
from pathlib import Path
from typing import Optional

from backend.config import get_config
from backend.models.base import ScamAnalysisProvider
from backend.schemas import (
    AnalysisResult,
    risk_level_from_score,
    verdict_from_score,
)

logger = logging.getLogger(__name__)


class LocalMLProvider(ScamAnalysisProvider):
    """
    Scam analysis provider using a locally trained TF-IDF + Logistic Regression model.

    Model artifacts must exist at:
        models/scam_classifier.joblib
        models/tfidf_vectorizer.joblib

    Run `python scripts/train_model.py` to generate these.
    """

    def __init__(self):
        self._classifier = None
        self._vectorizer = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._try_load_models()

    def _try_load_models(self):
        """Attempt to load the trained model artifacts."""
        try:
            import joblib

            config = get_config()
            model_cfg = config.get("model", {})
            model_dir = Path(__file__).parent.parent.parent / "models"

            classifier_path = model_dir / model_cfg.get("classifier_file", "scam_classifier.joblib")
            vectorizer_path = model_dir / model_cfg.get("vectorizer_file", "tfidf_vectorizer.joblib")

            if not classifier_path.exists():
                self._init_error = (
                    f"Model file not found: {classifier_path}. "
                    "Run `python scripts/train_model.py` to train the model."
                )
                return

            if not vectorizer_path.exists():
                self._init_error = (
                    f"Vectorizer file not found: {vectorizer_path}. "
                    "Run `python scripts/train_model.py` to train the model."
                )
                return

            self._classifier = joblib.load(str(classifier_path))
            self._vectorizer = joblib.load(str(vectorizer_path))
            self._initialized = True
            logger.info("LocalMLProvider: models loaded successfully.")

        except ImportError:
            self._init_error = "joblib/scikit-learn not installed."
        except Exception as e:
            self._init_error = f"Model loading failed: {str(e)}"
            logger.error(self._init_error)

    def is_available(self) -> bool:
        return self._initialized

    def provider_name(self) -> str:
        return "Local ML (TF-IDF + Logistic Regression)"

    def analyze_text(self, text: str) -> AnalysisResult:
        """
        Predict scam probability using the local ML model.

        Returns an AnalysisResult with a probability-based risk score.
        The local model provides a numerical signal; detailed indicators
        come from the rule engine and/or Gemini.
        """
        if not self.is_available():
            raise RuntimeError(
                f"LocalMLProvider not available: {self._init_error}"
            )

        try:
            # Transform text with TF-IDF
            features = self._vectorizer.transform([text])

            # Get class probabilities
            proba = self._classifier.predict_proba(features)[0]

            # Find the probability of the scam class
            classes = list(self._classifier.classes_)
            if 1 in classes:
                scam_prob = proba[classes.index(1)]
            elif "scam" in classes:
                scam_prob = proba[classes.index("scam")]
            else:
                # Assume last class is the positive (scam) class
                scam_prob = proba[-1]

            scam_prob = float(max(0.0, min(1.0, scam_prob)))
            risk_score = scam_prob * 100.0

            return AnalysisResult(
                risk_score=risk_score,
                risk_level=risk_level_from_score(risk_score),
                verdict=verdict_from_score(risk_score),
                confidence=scam_prob,
                explanation=(
                    f"Local ML model predicts {scam_prob:.1%} probability of this being a scam. "
                    "This score is based on patterns learned from the training dataset."
                ),
                recommendation=(
                    "Review the warning indicators and verify the opportunity independently."
                ),
                provider_used="local_ml",
            )

        except Exception as e:
            logger.error(f"LocalMLProvider.analyze_text failed: {e}")
            raise
