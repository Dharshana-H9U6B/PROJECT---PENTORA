# -*- coding: utf-8 -*-
"""
tests/test_gemini_diagnostic.py
ScamCheck — Gemini Integration Diagnostic Tests

Covers:
  A. Text → Gemini → structured AnalysisResult
  B. Image → Gemini → structured AnalysisResult
  C. Gemini failure → controlled fallback (no crash)
  D. API key is never printed or exposed

Run only these tests:
  python -m pytest tests/test_gemini_diagnostic.py -v -s

Skip network tests (offline / CI):
  python -m pytest tests/test_gemini_diagnostic.py -v -m "not integration"
"""

import logging
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image as PILImage

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.schemas import AnalysisResult, RiskLevel, Verdict
from backend.models.gemini_provider import GeminiProvider
from backend.services.analysis_service import AnalysisService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SCAM_TEXT = (
    "Analyze this sentence for recruitment scam indicators: "
    "Pay ₹2999 registration fee immediately to secure your internship."
)


def _make_synthetic_image() -> PILImage.Image:
    """Create a small synthetic 'screenshot' image for testing."""
    return PILImage.new("RGB", (200, 100), color=(240, 240, 240))


def _assert_valid_result(result: AnalysisResult, *, label: str = ""):
    """Assert that result conforms to the normalized AnalysisResult schema."""
    assert isinstance(result, AnalysisResult), f"{label}: Expected AnalysisResult, got {type(result)}"
    assert 0.0 <= result.risk_score <= 100.0, f"{label}: risk_score={result.risk_score} out of [0,100]"
    assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"), f"{label}: invalid risk_level={result.risk_level!r}"
    assert result.verdict in ("LIKELY_LEGIT", "SUSPICIOUS", "POTENTIAL_SCAM", "HIGH_RISK_SCAM"), f"{label}: invalid verdict={result.verdict!r}"
    assert 0.0 <= result.confidence <= 1.0, f"{label}: confidence={result.confidence} out of [0,1]"
    assert isinstance(result.explanation, str), f"{label}: explanation must be str"
    assert isinstance(result.recommendation, str), f"{label}: recommendation must be str"
    assert isinstance(result.warning_indicators, list), f"{label}: warning_indicators must be list"


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — no network required
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiProviderSchema:
    """Schema and contract tests — mocked, no real API calls."""

    def test_validate_structured_result_shape(self):
        """AnalysisResult dataclass has all required fields."""
        r = AnalysisResult(
            risk_score=85.0,
            risk_level="CRITICAL",
            verdict="HIGH_RISK_SCAM",
            confidence=0.92,
            explanation="Test explanation.",
            recommendation="Do not engage.",
            provider_used="gemini",
        )
        _assert_valid_result(r, label="direct_construct")

    def test_provider_unavailable_no_crash(self):
        """If API key is absent, GeminiProvider initializes without crashing."""
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value=None):
            provider = GeminiProvider()
        assert not provider.is_available()
        assert provider._init_error is not None
        assert "GEMINI_API_KEY" in provider._init_error

    def test_analyze_text_raises_when_unavailable(self):
        """analyze_text raises RuntimeError (not swallows) when provider is not available."""
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value=None):
            provider = GeminiProvider()
        with pytest.raises(RuntimeError, match="GeminiProvider not available"):
            provider.analyze_text("some text")

    def test_analyze_image_raises_when_unavailable(self):
        """analyze_image raises RuntimeError (not swallows) when provider is not available."""
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value=None):
            provider = GeminiProvider()
        img = _make_synthetic_image()
        with pytest.raises(RuntimeError, match="GeminiProvider not available"):
            provider.analyze_image(img)

    def test_analyze_service_image_fallback_no_crash(self):
        """AnalysisService.analyze_image returns safe AnalysisResult if Gemini is down."""
        svc = AnalysisService()
        # Force gemini to appear unavailable
        with patch.object(svc, "_gemini", None):
            svc._initialized = True
            result = svc.analyze_image(_make_synthetic_image())
        assert isinstance(result, AnalysisResult)
        assert result.provider_used == "unavailable"
        assert result.analysis_error is not None
        assert "Gemini unavailable" in result.analysis_error

    def test_gemini_error_fallback_returns_structured_result(self):
        """AnalysisService returns a structured result when Gemini raises mid-call."""
        svc = AnalysisService()
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.analyze_image.side_effect = ConnectionError("Simulated network error")
        mock_provider._model_name = "gemini-3.6-flash"
        svc._gemini = mock_provider
        svc._ml = None
        svc._initialized = True

        result = svc.analyze_image(_make_synthetic_image())
        assert isinstance(result, AnalysisResult)
        assert result.provider_used == "unavailable"
        # Error message should reference the exception type
        assert result.analysis_error is not None

    def test_is_retryable_detects_win_error(self):
        """_is_retryable correctly identifies WinError 10053."""
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value="fake-key"):
            with patch("backend.models.gemini_provider.genai" if False else "google.genai.Client"):
                provider = GeminiProvider.__new__(GeminiProvider)
                provider._client = MagicMock()
                provider._model_name = "gemini-3.6-flash"
                provider._initialized = True
                provider._init_error = None

        exc = Exception("[WinError 10053] An established connection was aborted")
        assert provider._is_retryable(exc) is True

        exc_ok = ValueError("invalid json")
        assert provider._is_retryable(exc_ok) is False


class TestSecurityNoKeyLeak:
    """Verify API key is never surfaced in logs or error messages."""

    def test_init_error_does_not_contain_api_key(self, caplog):
        """Init error message must not include the actual API key."""
        fake_key = "FAKE-SECRET-KEY-12345"
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value=fake_key):
            with patch("google.genai.Client", side_effect=RuntimeError("test init error")):
                with caplog.at_level(logging.ERROR):
                    provider = GeminiProvider()

        # Key must never appear in logs
        for record in caplog.records:
            assert fake_key not in record.message, "API key leaked into log message!"
        # Key must never appear in init error string
        assert fake_key not in (provider._init_error or ""), "API key leaked into _init_error!"

    def test_api_key_presence_logged_as_boolean_only(self, caplog):
        """Logging must only reveal True/False for API key, never the value."""
        fake_key = "SUPER-SECRET-API-KEY"
        with patch("backend.models.gemini_provider.get_gemini_api_key", return_value=fake_key):
            with patch("google.genai.Client", return_value=MagicMock()):
                with caplog.at_level(logging.INFO):
                    provider = GeminiProvider()

        for record in caplog.records:
            assert fake_key not in record.message, f"API key leaked in log: {record.message}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — require real network + valid GEMINI_API_KEY in .env
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestGeminiIntegrationLive:
    """
    Live API integration tests.
    These require GEMINI_API_KEY to be set and network access.
    Skipped automatically if provider is not available.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_key(self):
        from backend.config import get_gemini_api_key
        if not get_gemini_api_key():
            pytest.skip("GEMINI_API_KEY not set — skipping live integration test")

    @pytest.fixture(autouse=True, name="provider")
    def _provider(self, skip_if_no_key):
        p = GeminiProvider()
        if not p.is_available():
            pytest.skip(f"GeminiProvider unavailable: {p._init_error}")
        return p

    def test_text_analysis_returns_structured_result(self, _provider):
        """A: Text → Gemini → structured AnalysisResult with valid schema."""
        result = _provider.analyze_text(SCAM_TEXT)
        _assert_valid_result(result, label="text_analysis")
        assert result.provider_used == "gemini"
        # The scam text contains a fee demand — expect elevated risk
        assert result.risk_score >= 30, (
            f"Expected risk_score >= 30 for fee-demand scam text, got {result.risk_score}"
        )

    def test_image_analysis_returns_structured_result(self, _provider):
        """B: Image → Gemini → structured AnalysisResult with valid schema."""
        img = _make_synthetic_image()
        result = _provider.analyze_image(img, context="Synthetic test image for automated testing.")
        _assert_valid_result(result, label="image_analysis")
        assert result.provider_used == "gemini"

    def test_service_text_analysis_end_to_end(self):
        """A (service-level): AnalysisService.analyze_text returns structured result."""
        svc = AnalysisService()
        result = svc.analyze_text(SCAM_TEXT)
        _assert_valid_result(result, label="service_text")
        assert "gemini" in result.provider_used.lower(), (
            f"Expected Gemini in provider_used, got: {result.provider_used!r}"
        )
        assert result.risk_score >= 30

    def test_service_image_analysis_end_to_end(self):
        """B (service-level): AnalysisService.analyze_image returns structured result."""
        svc = AnalysisService()
        img = _make_synthetic_image()
        result = svc.analyze_image(img, context="Automated integration test.")
        _assert_valid_result(result, label="service_image")
        # Either gemini succeeded or fallback returned clean result
        assert result.provider_used in ("gemini", "unavailable")

    def test_bad_model_name_propagates_exception(self):
        """C: Gemini failure with bad model → exception propagates, no silent swallow."""
        provider = GeminiProvider()
        if not provider.is_available():
            pytest.skip("Provider unavailable")
        provider._model_name = "gemini-NONEXISTENT-9999"
        with pytest.raises(Exception) as exc_info:
            provider.analyze_text(SCAM_TEXT)
        err_msg = str(exc_info.value)
        # Should be a real API error, not a silent empty result
        assert len(err_msg) > 0
        assert exc_info.type.__name__ != "AssertionError"
