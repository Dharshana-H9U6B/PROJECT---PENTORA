"""
Analysis Service — the primary entry point for all scam analysis requests.

The Streamlit UI calls ONLY this service.
It orchestrates providers, the risk engine, and input processors.
"""

import logging
from typing import Optional
from PIL.Image import Image as PILImage

from backend.schemas import AnalysisResult
from backend.models.gemini_provider import GeminiProvider
from backend.models.local_ml_provider import LocalMLProvider
from backend.risk_engine import build_final_result
from backend.input_processors.text_processor import clean_text, build_structured_text
from backend.input_processors.image_processor import preprocess_image

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Orchestrates all analysis providers and the risk engine.

    The Streamlit UI interacts only with this class.
    Provider initialization is lazy and failure-tolerant.
    """

    def __init__(self):
        self._gemini: Optional[GeminiProvider] = None
        self._ml: Optional[LocalMLProvider] = None
        self._initialized = False
        # Tracks the last runtime error from Gemini (separate from init errors)
        self._last_gemini_error: Optional[str] = None

    def _ensure_initialized(self):
        """Lazy initialization of providers."""
        if self._initialized:
            return

        # Initialize Gemini
        try:
            self._gemini = GeminiProvider()
            if self._gemini.is_available():
                logger.info(
                    "GeminiProvider ready. model=%s, api_key_present=%s",
                    self._gemini._model_name,
                    self._gemini._client is not None,
                )
            else:
                logger.warning(
                    "GeminiProvider unavailable: %s", self._gemini._init_error
                )
        except Exception as e:
            logger.error("Failed to create GeminiProvider: %s", e)
            self._gemini = None

        # Initialize Local ML
        try:
            self._ml = LocalMLProvider()
            if self._ml.is_available():
                logger.info("LocalMLProvider ready.")
            else:
                logger.info(
                    "LocalMLProvider not trained yet: %s", self._ml._init_error
                )
        except Exception as e:
            logger.error("Failed to create LocalMLProvider: %s", e)
            self._ml = None

        self._initialized = True

    def get_provider_status(self) -> dict:
        """Return availability status of each provider."""
        self._ensure_initialized()
        gemini_ok = self._gemini is not None and self._gemini.is_available()
        gemini_err = None
        if not gemini_ok:
            gemini_err = (
                getattr(self._gemini, "_init_error", None)
                or "Not initialized"
            ) if self._gemini else "Not initialized"
        if self._last_gemini_error:
            gemini_err = self._last_gemini_error

        return {
            "gemini": {
                "available": gemini_ok,
                "name": self._gemini.provider_name() if self._gemini else "Gemini",
                "error": gemini_err,
            },
            "local_ml": {
                "available": self._ml is not None and self._ml.is_available(),
                "name": self._ml.provider_name() if self._ml else "Local ML",
                "error": getattr(self._ml, "_init_error", None) if self._ml else "Not initialized",
            },
        }

    def analyze_text(self, text: str) -> AnalysisResult:
        """
        Analyze a raw text message.

        Args:
            text: Opportunity message/description (raw input).

        Returns:
            AnalysisResult with combined risk assessment.
        """
        self._ensure_initialized()

        # Clean input
        cleaned_text = clean_text(text)

        # Run Gemini
        gemini_result = None
        if self._gemini and self._gemini.is_available():
            try:
                logger.info(f"[AnalysisService:analyze_text] Starting Gemini analysis. model={self._gemini._model_name}")
                gemini_result = self._gemini.analyze_text(cleaned_text)
                self._last_gemini_error = None
                logger.info(f"[AnalysisService:analyze_text] Gemini analysis succeeded. score={gemini_result.risk_score}")
            except Exception as e:
                self._last_gemini_error = f"{type(e).__name__}: {e}"
                logger.error(f"[AnalysisService:analyze_text] [Exception: {type(e).__name__}] Gemini text analysis failed: {e}")
        else:
            logger.warning("[AnalysisService:analyze_text] Gemini is not available.")

        # Run Local ML
        ml_result = None
        if self._ml and self._ml.is_available():
            try:
                ml_result = self._ml.analyze_text(cleaned_text)
                logger.info(f"[AnalysisService:analyze_text] Local ML analysis succeeded: score={ml_result.risk_score}")
            except Exception as e:
                logger.error(f"[AnalysisService:analyze_text] [Exception: {type(e).__name__}] Local ML analysis failed: {e}")

        # Check if at least one provider worked
        if gemini_result is None and ml_result is None:
            # Rules-only fallback
            logger.warning("[AnalysisService:analyze_text] No AI providers available, using rules only.")

        # Build final result
        final = build_final_result(cleaned_text, gemini_result, ml_result)
        return final

    def analyze_structured(
        self,
        company: str = "",
        role: str = "",
        salary: str = "",
        registration_fee: str = "",
        contact_method: str = "",
        website: str = "",
        description: str = "",
    ) -> AnalysisResult:
        """
        Analyze a structured job/internship form.
        Converts to text and delegates to the appropriate providers.
        """
        self._ensure_initialized()

        # Build normalized text
        normalized_text = build_structured_text(
            company=company,
            role=role,
            salary=salary,
            registration_fee=registration_fee,
            contact_method=contact_method,
            website=website,
            description=description,
        )

        # Run Gemini with structured prompt if available
        gemini_result = None
        if self._gemini and self._gemini.is_available():
            try:
                logger.info(f"[AnalysisService:analyze_structured] Starting Gemini analysis. model={self._gemini._model_name}")
                gemini_result = self._gemini.analyze_structured(
                    company=company,
                    role=role,
                    salary=salary,
                    registration_fee=registration_fee,
                    contact_method=contact_method,
                    website=website,
                    description=description,
                )
                self._last_gemini_error = None
                logger.info(f"[AnalysisService:analyze_structured] Gemini analysis succeeded. score={gemini_result.risk_score}")
            except Exception as e:
                self._last_gemini_error = f"{type(e).__name__}: {e}"
                logger.error(f"[AnalysisService:analyze_structured] [Exception: {type(e).__name__}] Gemini structured analysis failed: {e}")

        # Run Local ML on normalized text
        ml_result = None
        if self._ml and self._ml.is_available():
            try:
                ml_result = self._ml.analyze_text(normalized_text)
                logger.info(f"[AnalysisService:analyze_structured] Local ML analysis succeeded: score={ml_result.risk_score}")
            except Exception as e:
                logger.error(f"[AnalysisService:analyze_structured] [Exception: {type(e).__name__}] Local ML structured analysis failed: {e}")

        return build_final_result(normalized_text, gemini_result, ml_result)

    def analyze_image(self, image: PILImage, context: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a screenshot image using Gemini's multimodal capabilities.

        Falls back to safe diagnostic result if Gemini is unavailable.
        """
        self._ensure_initialized()

        # Preprocess image
        processed_image = preprocess_image(image)

        # Gemini multimodal analysis
        gemini_result = None
        if self._gemini and self._gemini.is_available():
            try:
                logger.info(f"[AnalysisService:analyze_image] Starting Gemini multimodal analysis. model={self._gemini._model_name}")
                gemini_result = self._gemini.analyze_image(processed_image, context)
                self._last_gemini_error = None
                logger.info(f"[AnalysisService:analyze_image] Gemini image analysis succeeded. score={gemini_result.risk_score}")
            except Exception as e:
                self._last_gemini_error = f"{type(e).__name__}: {e}"
                logger.error(f"[AnalysisService:analyze_image] [Exception: {type(e).__name__}] Gemini image analysis failed: {e}")
        else:
            init_err = getattr(self._gemini, "_init_error", None) if self._gemini else "GeminiProvider not initialized"
            self._last_gemini_error = init_err
            logger.warning(f"[AnalysisService:analyze_image] Gemini is not available: {init_err}")

        if gemini_result is None:
            # Cannot do image analysis without Gemini (no local OCR)
            from backend.schemas import AnalysisResult, RiskLevel, Verdict
            diag_err = self._last_gemini_error or "Gemini provider unavailable"
            return AnalysisResult(
                risk_score=0,
                risk_level=RiskLevel.LOW,
                verdict=Verdict.LIKELY_LEGIT,
                confidence=0.0,
                explanation=(
                    "Image analysis requires Gemini API, which is currently unavailable. "
                    "Please configure your GEMINI_API_KEY or paste the message text instead."
                ),
                recommendation="Use the 'Paste Message' tab to analyze text directly.",
                provider_used="unavailable",
                analysis_error=f"Gemini unavailable for image analysis ({diag_err}).",
            )

        # For images, we trust Gemini's result
        return gemini_result


# Singleton instance
_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Return the singleton AnalysisService."""
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service
