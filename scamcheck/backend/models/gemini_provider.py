"""
Gemini provider implementation.

Uses the Google GenAI Python SDK to call Gemini for scam analysis.
All prompt logic is imported from backend/prompts/.
"""

import json
import logging
import re
import time
from typing import Optional, Callable, TypeVar

_T = TypeVar("_T")

# Transient network errors that are safe to retry
_RETRYABLE_MESSAGES = (
    "WinError 10053",  # Connection aborted by host
    "WinError 10054",  # Connection reset by remote
    "ReadError",
    "ConnectError",
    "RemoteProtocolError",
    "Connection reset",
)

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
        logger.info("[GeminiProvider:init] Starting Gemini provider initialization.")
        try:
            from google import genai

            api_key = get_gemini_api_key()
            has_key = bool(api_key and len(api_key.strip()) > 0)
            logger.info(f"[GeminiProvider:init] api_key_present={has_key}")

            if not has_key:
                self._init_error = "GEMINI_API_KEY not set in environment."
                logger.warning(f"[GeminiProvider:init] {self._init_error}")
                return

            config = get_config()
            self._model_name = config.get("gemini", {}).get("model", "gemini-3.6-flash")
            logger.info(f"[GeminiProvider:init] configured_model={self._model_name}")

            self._client = genai.Client(api_key=api_key)
            self._initialized = True
            logger.info(f"[GeminiProvider:init] Successfully initialized with model: {self._model_name}")

        except ImportError as e:
            self._init_error = "google-genai package not installed."
            logger.error(f"[GeminiProvider:init] [Exception: {type(e).__name__}] {self._init_error}: {e}")
        except Exception as e:
            self._init_error = f"Gemini initialization failed ({type(e).__name__}): {str(e)}"
            logger.error(f"[GeminiProvider:init] [Exception: {type(e).__name__}] {self._init_error}")

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

    def _is_retryable(self, exc: Exception) -> bool:
        """Return True if exception is a transient network error worth retrying."""
        exc_str = str(exc)
        exc_type = type(exc).__name__
        return (
            any(msg in exc_str for msg in _RETRYABLE_MESSAGES)
            or any(t in exc_type for t in ("ReadError", "ConnectError", "RemoteProtocolError"))
        )

    def _call_with_retry(self, fn: Callable[[], _T], label: str, max_retries: int = 3) -> _T:
        """Call fn() with retry on transient network errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if self._is_retryable(e) and attempt < max_retries:
                    wait = 2 ** (attempt - 1)  # 1s, 2s backoff
                    logger.warning(
                        f"[GeminiProvider:{label}] Transient network error (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s. [{type(e).__name__}]: {e}"
                    )
                    time.sleep(wait)
                    last_exc = e
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    def _call_gemini(self, prompt: str) -> str:
        """Send a text prompt to Gemini and return the response text."""
        from google.genai import types

        logger.info(f"[GeminiProvider:text_call_start] model={self._model_name}")

        def _do_call() -> str:
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

        text = self._call_with_retry(_do_call, label="text_call")
        logger.info(f"[GeminiProvider:text_call_success] response length={len(text) if text else 0}")
        return text

    def _call_gemini_with_image(self, prompt: str, image: PILImage) -> str:
        """Send a multimodal (text + image) prompt to Gemini."""
        from google.genai import types

        logger.info(
            f"[GeminiProvider:image_call_start] model={self._model_name}, "
            f"image_size={image.size if hasattr(image, 'size') else 'unknown'}"
        )

        def _do_call() -> str:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    image,
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            return response.text

        text = self._call_with_retry(_do_call, label="image_call")
        logger.info(f"[GeminiProvider:image_call_success] response length={len(text) if text else 0}")
        return text

    def _parse_response(self, raw_text: str) -> AnalysisResult:
        """Parse and validate Gemini's JSON response."""
        logger.info("[GeminiProvider:parse_start] Parsing raw response")
        try:
            data = self._extract_json(raw_text)
            result = validate_gemini_response(data)
            logger.info(f"[GeminiProvider:parse_success] parsed risk_score={result.risk_score}, risk_level={result.risk_level}")
            return result
        except Exception as e:
            logger.error(f"[GeminiProvider:parse_error] [Exception: {type(e).__name__}] Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response preview: {raw_text[:500]}")
            # Return a safe fallback result
            return AnalysisResult(
                risk_score=50.0,
                risk_level="MEDIUM",
                verdict="SUSPICIOUS",
                confidence=0.3,
                explanation="Analysis partially completed. Could not fully parse AI response.",
                recommendation="Please verify this opportunity independently.",
                provider_used="gemini",
                analysis_error=f"Response parsing error ({type(e).__name__}): {str(e)}",
            )

    def analyze_text(self, text: str) -> AnalysisResult:
        """Analyze a text message using Gemini."""
        if not self.is_available():
            err_msg = f"GeminiProvider not available: {self._init_error}"
            logger.error(f"[GeminiProvider:analyze_text] {err_msg}")
            raise RuntimeError(err_msg)

        prompt = TEXT_ANALYSIS_PROMPT.format(text=text)
        try:
            raw = self._call_gemini(prompt)
            result = self._parse_response(raw)
            result.provider_used = "gemini"
            return result
        except Exception as e:
            logger.error(f"[GeminiProvider:analyze_text] [Exception: {type(e).__name__}] Gemini analyze_text failed: {e}")
            raise

    def analyze_image(self, image: PILImage, context: Optional[str] = None) -> AnalysisResult:
        """Analyze a screenshot using Gemini's multimodal capabilities."""
        if not self.is_available():
            err_msg = f"GeminiProvider not available: {self._init_error}"
            logger.error(f"[GeminiProvider:analyze_image] {err_msg}")
            raise RuntimeError(err_msg)

        prompt = IMAGE_ANALYSIS_PROMPT
        if context:
            prompt += f"\n\nADDITIONAL CONTEXT: {context}"

        try:
            raw = self._call_gemini_with_image(prompt, image)
            result = self._parse_response(raw)
            result.provider_used = "gemini"
            return result
        except Exception as e:
            logger.error(f"[GeminiProvider:analyze_image] [Exception: {type(e).__name__}] Gemini analyze_image failed: {e}")
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
            err_msg = f"GeminiProvider not available: {self._init_error}"
            logger.error(f"[GeminiProvider:analyze_structured] {err_msg}")
            raise RuntimeError(err_msg)

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
            logger.error(f"[GeminiProvider:analyze_structured] [Exception: {type(e).__name__}] Gemini analyze_structured failed: {e}")
            raise
