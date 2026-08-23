"""
Gemini provider implementation.

Uses the Google GenAI Python SDK to call Gemini for scam analysis.
All prompt logic is imported from backend/prompts/.
"""

import json
import logging
import re
from typing import Optional

from PIL.Image import Image as PILImage

from backend.config import get_config, get_gemini_api_key
from backend.models.base import ScamAnalysisProvider
from backend.prompts.scam_analysis_prompt import (
    SYSTEM_PROMPT,
    TEXT_ANALYSIS_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
    STRUCTURED_ANALYSIS_PROMPT,
)
from backend.schemas import (
    AnalysisResult,
    WarningIndicator,
    validate_gemini_response,
    risk_level_from_score,
    verdict_from_score,
)

logger = logging.getLogger(__name__)


class GeminiProvider(ScamAnalysisProvider):
    """
    Scam analysis provider using Google Gemini API.

    Requires GEMINI_API_KEY environment variable.
    Model is configurable via GEMINI_MODEL env var or config.yaml.
    """

    def __init__(self):
        self._client = None
        self._model_name: Optional[str] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._try_init()

    def _try_init(self):
        """Attempt to initialize the Gemini client."""
        try:
            from google import genai

            api_key = get_gemini_api_key()
            if not api_key:
                self._init_error = "GEMINI_API_KEY not set in environment."
                return

            config = get_config()
            self._model_name = config.get("gemini", {}).get("model", "gemini-2.0-flash")

            self._client = genai.Client(api_key=api_key)
            self._initialized = True
            logger.info(f"GeminiProvider initialized with model: {self._model_name}")

        except ImportError:
            self._init_error = "google-genai package not installed."
        except Exception as e:
            self._init_error = f"Gemini initialization failed: {str(e)}"
            logger.error(self._init_error)

    def is_available(self) -> bool:
        return self._initialized and self._client is not None

    def provider_name(self) -> str:
        return f"Gemini ({self._model_name})"

    def _extract_json(self, text: str) -> dict:
        """
        Attempt to extract and parse JSON from Gemini's response.
        Handles cases where the model wraps JSON in markdown code fences.
        """
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` or ``` ... ```
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")

    def _call_gemini(self, prompt: str) -> str:
        """Send a text prompt to Gemini and return the response text."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        return response.text

    def _call_gemini_with_image(self, prompt: str, image: PILImage) -> str:
        """Send a multimodal (text + image) prompt to Gemini."""
        from google.genai import types

        # Convert PIL image to bytes for the API
        import io
        img_bytes = io.BytesIO()
        fmt = image.format or "PNG"
        image.save(img_bytes, format=fmt)
        img_bytes.seek(0)

        mime_type = f"image/{fmt.lower()}"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=img_bytes.read(), mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        return response.text

    def _parse_response(self, raw_text: str) -> AnalysisResult:
        """Parse and validate Gemini's JSON response."""
        try:
            data = self._extract_json(raw_text)
            return validate_gemini_response(data)
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {raw_text[:500]}")
            # Return a safe fallback result
            return AnalysisResult(
                risk_score=50.0,
                risk_level="MEDIUM",
                verdict="SUSPICIOUS",
                confidence=0.3,
                explanation="Analysis partially completed. Could not fully parse AI response.",
                recommendation="Please verify this opportunity independently.",
                provider_used="gemini",
                analysis_error=f"Response parsing error: {str(e)}",
            )

    def analyze_text(self, text: str) -> AnalysisResult:
        """Analyze a text message using Gemini."""
        if not self.is_available():
            raise RuntimeError(
                f"GeminiProvider not available: {self._init_error}"
            )

        prompt = TEXT_ANALYSIS_PROMPT.format(text=text)
        try:
            raw = self._call_gemini(prompt)
            result = self._parse_response(raw)
            result.provider_used = "gemini"
            return result
        except Exception as e:
            logger.error(f"Gemini analyze_text failed: {e}")
            raise

    def analyze_image(self, image: PILImage, context: Optional[str] = None) -> AnalysisResult:
        """Analyze a screenshot using Gemini's multimodal capabilities."""
        if not self.is_available():
            raise RuntimeError(
                f"GeminiProvider not available: {self._init_error}"
            )

        prompt = IMAGE_ANALYSIS_PROMPT
        if context:
            prompt += f"\n\nADDITIONAL CONTEXT: {context}"

        try:
            raw = self._call_gemini_with_image(prompt, image)
            result = self._parse_response(raw)
            result.provider_used = "gemini"
            return result
        except Exception as e:
            logger.error(f"Gemini analyze_image failed: {e}")
            raise

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
        """Analyze a structured job/internship opportunity form."""
        if not self.is_available():
            raise RuntimeError(
                f"GeminiProvider not available: {self._init_error}"
            )

        prompt = STRUCTURED_ANALYSIS_PROMPT.format(
            company=company or "Not provided",
            role=role or "Not provided",
            salary=salary or "Not provided",
            registration_fee=registration_fee or "None mentioned",
            contact_method=contact_method or "Not provided",
            website=website or "Not provided",
            description=description or "Not provided",
        )
        try:
            raw = self._call_gemini(prompt)
            result = self._parse_response(raw)
            result.provider_used = "gemini"
            return result
        except Exception as e:
            logger.error(f"Gemini analyze_structured failed: {e}")
            raise
