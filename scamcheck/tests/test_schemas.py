"""
Tests for schema validation and Gemini response parsing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.schemas import (
    validate_gemini_response,
    risk_level_from_score,
    verdict_from_score,
    AnalysisResult,
    RiskLevel,
    Verdict,
)


class TestRiskLevelFromScore:
    def test_low(self):
        assert risk_level_from_score(10) == RiskLevel.LOW
        assert risk_level_from_score(24) == RiskLevel.LOW

    def test_medium(self):
        assert risk_level_from_score(25) == RiskLevel.MEDIUM
        assert risk_level_from_score(74) == RiskLevel.MEDIUM

    def test_high(self):
        assert risk_level_from_score(75) == RiskLevel.HIGH
        assert risk_level_from_score(89) == RiskLevel.HIGH

    def test_critical(self):
        assert risk_level_from_score(90) == RiskLevel.CRITICAL
        assert risk_level_from_score(100) == RiskLevel.CRITICAL


class TestVerdictFromScore:
    def test_likely_legit(self):
        assert verdict_from_score(10) == Verdict.LIKELY_LEGIT

    def test_suspicious(self):
        assert verdict_from_score(50) == Verdict.SUSPICIOUS

    def test_potential_scam(self):
        assert verdict_from_score(75) == Verdict.POTENTIAL_SCAM

    def test_high_risk_scam(self):
        assert verdict_from_score(92) == Verdict.HIGH_RISK_SCAM


class TestValidateGeminiResponse:
    def test_valid_response(self):
        data = {
            "risk_score": 85,
            "risk_level": "HIGH",
            "verdict": "POTENTIAL_SCAM",
            "confidence": 0.92,
            "warning_indicators": [
                {
                    "type": "UPFRONT_PAYMENT",
                    "severity": "HIGH",
                    "evidence": "Pay ₹2999 registration fee",
                    "description": "Upfront payment requested",
                }
            ],
            "explanation": "Multiple red flags detected.",
            "recommendation": "Do not pay.",
            "evidence": ["Registration fee requested"],
        }
        result = validate_gemini_response(data)
        assert isinstance(result, AnalysisResult)
        assert result.risk_score == 85.0
        assert result.risk_level == "HIGH"
        assert result.verdict == "POTENTIAL_SCAM"
        assert result.confidence == 0.92
        assert len(result.warning_indicators) == 1
        assert result.warning_indicators[0].type == "UPFRONT_PAYMENT"

    def test_out_of_bounds_score_clipped(self):
        data = {"risk_score": 150, "confidence": 2.0}
        result = validate_gemini_response(data)
        assert result.risk_score == 100.0
        assert result.confidence == 1.0

    def test_invalid_risk_level_falls_back(self):
        data = {"risk_score": 80, "risk_level": "INVALID_LEVEL"}
        result = validate_gemini_response(data)
        # Should fall back to derived level from score
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_missing_fields_get_defaults(self):
        data = {}
        result = validate_gemini_response(data)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.risk_score <= 100.0

    def test_malformed_indicators_skipped(self):
        data = {
            "risk_score": 60,
            "warning_indicators": [
                "this is not a dict",
                123,
                {"type": "VALID", "severity": "HIGH", "evidence": "test"},
            ],
        }
        result = validate_gemini_response(data)
        # Only the valid dict indicator should be parsed
        assert len(result.warning_indicators) == 1
