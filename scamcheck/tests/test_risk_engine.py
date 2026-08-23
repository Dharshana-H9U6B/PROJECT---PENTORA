"""
Tests for the risk engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.risk_engine import calculate_final_risk, run_rule_engine, build_final_result
from backend.schemas import AnalysisResult, RiskLevel


class TestRiskScoreCalculation:
    def test_gemini_only_score(self):
        """When only Gemini is available, it should carry most weight."""
        score = calculate_final_risk(
            ml_score=None,
            gemini_score=80.0,
            rule_score=40.0,
        )
        assert 0.0 <= score <= 100.0
        # Should be above 60 given Gemini says 80
        assert score > 55.0

    def test_rules_only_score(self):
        """When only rules are available, the rule score is used."""
        score = calculate_final_risk(
            ml_score=None,
            gemini_score=None,
            rule_score=70.0,
        )
        assert score == 70.0

    def test_combined_high_risk_score(self):
        """All providers agreeing on high risk should produce high final score."""
        score = calculate_final_risk(
            ml_score=85.0,
            gemini_score=90.0,
            rule_score=75.0,
        )
        assert score >= 75.0

    def test_combined_low_risk_score(self):
        """All providers agreeing on low risk should produce low final score."""
        score = calculate_final_risk(
            ml_score=10.0,
            gemini_score=5.0,
            rule_score=0.0,
        )
        assert score < 20.0

    def test_score_bounds(self):
        """Score must always be within 0–100."""
        score = calculate_final_risk(
            ml_score=150.0,  # Out of bounds input
            gemini_score=200.0,
            rule_score=300.0,
        )
        assert 0.0 <= score <= 100.0

    def test_custom_weights(self):
        """Custom weights should be respected."""
        # With 100% weight on rules
        score = calculate_final_risk(
            ml_score=50.0,
            gemini_score=50.0,
            rule_score=90.0,
            weights={"ml": 0.0, "gemini": 0.0, "rules": 1.0},
        )
        assert abs(score - 90.0) < 1.0


class TestRuleEngine:
    def test_high_risk_message_produces_indicators(self):
        text = "Pay ₹2999 registration fee now. Only 3 seats left. WhatsApp only."
        indicators, score = run_rule_engine(text)
        assert len(indicators) > 0
        assert score > 0.0

    def test_clean_message_produces_low_score(self):
        text = "Your application has been reviewed. Please attend the scheduled interview at our office."
        indicators, score = run_rule_engine(text)
        # Low scam signals = low rule score
        assert score < 30.0

    def test_score_caps_at_100(self):
        """Extremely suspicious text should cap at 100."""
        text = (
            "PAY ₹5000 REGISTRATION FEE NOW! Only 1 seat left! Act now! "
            "Share OTP and bank account! Guaranteed job! No interview needed! "
            "WhatsApp only! Limited time! UPI PIN required! Security deposit!"
        )
        indicators, score = run_rule_engine(text)
        assert score <= 100.0


class TestBuildFinalResult:
    def test_no_providers_uses_rules_only(self):
        text = "Pay ₹999 to confirm your internship position. Act immediately!"
        result = build_final_result(text, gemini_result=None, ml_result=None)
        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.risk_score <= 100.0
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_gemini_result_incorporated(self):
        gemini = AnalysisResult(
            risk_score=85.0,
            risk_level="HIGH",
            verdict="POTENTIAL_SCAM",
            confidence=0.9,
            explanation="Test explanation",
            recommendation="Test recommendation",
        )
        text = "Pay ₹500 registration fee now."
        result = build_final_result(text, gemini_result=gemini, ml_result=None)
        assert result.risk_score > 50.0
        assert "gemini" in result.provider_used
