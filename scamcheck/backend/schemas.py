"""
Internal data schemas for ScamCheck analysis results.
All providers must normalize their output into AnalysisResult.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Verdict(str, Enum):
    LIKELY_LEGIT = "LIKELY_LEGIT"
    SUSPICIOUS = "SUSPICIOUS"
    POTENTIAL_SCAM = "POTENTIAL_SCAM"
    HIGH_RISK_SCAM = "HIGH_RISK_SCAM"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class WarningIndicator:
    type: str
    severity: str
    evidence: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "evidence": self.evidence,
            "description": self.description,
        }


@dataclass
class AnalysisResult:
    """
    Normalized analysis result from any provider.
    This is the single internal schema used throughout the application.
    """
    risk_score: float = 0.0            # 0–100
    risk_level: str = RiskLevel.LOW    # LOW / MEDIUM / HIGH / CRITICAL
    verdict: str = Verdict.LIKELY_LEGIT
    confidence: float = 0.0            # 0.0–1.0
    warning_indicators: List[WarningIndicator] = field(default_factory=list)
    explanation: str = ""
    recommendation: str = ""
    evidence: List[str] = field(default_factory=list)

    # Score breakdown (optional, for display)
    ml_score: Optional[float] = None
    gemini_score: Optional[float] = None
    rule_score: Optional[float] = None

    # Provider info
    provider_used: str = "unknown"
    analysis_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "warning_indicators": [w.to_dict() for w in self.warning_indicators],
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "ml_score": self.ml_score,
            "gemini_score": self.gemini_score,
            "rule_score": self.rule_score,
            "provider_used": self.provider_used,
            "analysis_error": self.analysis_error,
        }


def risk_level_from_score(score: float) -> str:
    """Derive risk level string from a 0–100 score."""
    if score >= 90:
        return RiskLevel.CRITICAL
    elif score >= 75:
        return RiskLevel.HIGH
    elif score >= 50:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def verdict_from_score(score: float) -> str:
    """Derive verdict string from a 0–100 score."""
    if score >= 90:
        return Verdict.HIGH_RISK_SCAM
    elif score >= 70:
        return Verdict.POTENTIAL_SCAM
    elif score >= 45:
        return Verdict.SUSPICIOUS
    else:
        return Verdict.LIKELY_LEGIT


def validate_gemini_response(data: dict) -> AnalysisResult:
    """
    Parse and validate a raw Gemini JSON response into an AnalysisResult.
    Applies safe defaults for missing or invalid fields.
    """
    risk_score = float(data.get("risk_score", 50))
    risk_score = max(0.0, min(100.0, risk_score))

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    risk_level = data.get("risk_level", risk_level_from_score(risk_score))
    if risk_level not in [r.value for r in RiskLevel]:
        risk_level = risk_level_from_score(risk_score)

    verdict = data.get("verdict", verdict_from_score(risk_score))
    if verdict not in [v.value for v in Verdict]:
        verdict = verdict_from_score(risk_score)

    raw_indicators = data.get("warning_indicators", [])
    indicators = []
    for raw in raw_indicators:
        if isinstance(raw, dict):
            indicators.append(WarningIndicator(
                type=raw.get("type", "UNKNOWN"),
                severity=raw.get("severity", "MEDIUM"),
                evidence=raw.get("evidence", ""),
                description=raw.get("description", ""),
            ))

    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    return AnalysisResult(
        risk_score=risk_score,
        risk_level=risk_level,
        verdict=verdict,
        confidence=confidence,
        warning_indicators=indicators,
        explanation=str(data.get("explanation", "")),
        recommendation=str(data.get("recommendation", "")),
        evidence=evidence,
        provider_used="gemini",
    )
