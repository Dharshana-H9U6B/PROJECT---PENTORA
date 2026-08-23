"""
Abstract base class for all ScamCheck analysis providers.

Any provider (Gemini, Local ML, HuggingFace, fine-tuned model, etc.)
must implement this interface.

The rest of the application only depends on this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL.Image import Image as PILImage

from backend.schemas import AnalysisResult


class ScamAnalysisProvider(ABC):
    """
    Abstract provider interface for scam analysis.

    Implementations:
        - GeminiProvider: uses Google Gemini API
        - LocalMLProvider: uses local TF-IDF + classifier
        - (future) HuggingFaceProvider, FineTunedProvider, etc.
    """

    @abstractmethod
    def analyze_text(self, text: str) -> AnalysisResult:
        """
        Analyze a raw text string for scam indicators.

        Args:
            text: The opportunity message/description to analyze.

        Returns:
            AnalysisResult with normalized scores and indicators.
        """
        raise NotImplementedError

    def analyze_image(self, image: PILImage, context: Optional[str] = None) -> AnalysisResult:
        """
        Analyze an image (screenshot) for scam indicators.

        Default implementation raises NotImplementedError.
        Override in multimodal-capable providers.

        Args:
            image: PIL Image object.
            context: Optional additional text context.

        Returns:
            AnalysisResult with normalized scores and indicators.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image analysis."
        )

    def is_available(self) -> bool:
        """
        Check whether this provider is available and configured correctly.

        Returns:
            True if the provider can be used, False otherwise.
        """
        return True

    def provider_name(self) -> str:
        """Return a human-readable name for this provider."""
        return self.__class__.__name__
