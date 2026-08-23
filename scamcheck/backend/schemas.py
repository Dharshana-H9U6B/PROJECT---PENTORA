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
    UNKNOWN = "UNKNOWN"      # Analysis failed — do not display as LOW


class Verdict(str, Enum):
    LIKELY_LEGIT = "LIKELY_LEGIT"
    SUSPICIOUS = "SUSPICIOUS"
    POTENTIAL_SCAM = "POTENTIAL_SCAM"
    HIGH_RISK_SCAM = "HIGH_RISK_SCAM"
    ANALYSIS_UNAVAILABLE = "ANALYSIS_UNAVAILABLE"   # Analysis failed


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OpportunityType(str, Enum):
    """Classification of the submitted opportunity."""
    JOB = "JOB"
    INTERNSHIP = "INTERNSHIP"
    PAID_TRAINING = "PAID_TRAINING"
    CERTIFICATION = "CERTIFICATION"
    SCHOLARSHIP = "SCHOLARSHIP"
    EVENT = "EVENT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class PaymentContext(str, Enum):
    """Why a payment is being requested, if any."""
    EMPLOYMENT_PAYMENT = "EMPLOYMENT_PAYMENT"           # Pay to get a job/internship
    TRAINING_PAYMENT = "TRAINING_PAYMENT"               # Pay for a course/training
    PRODUCT_OR_SERVICE_PAYMENT = "PRODUCT_OR_SERVICE_PAYMENT"
    APPLICATION_PAYMENT = "APPLICATION_PAYMENT"         # Pay to apply
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class PaymentRequiredFor(str, Enum):
    """What the payment is explicitly required for."""
    JOB_ACCESS = "JOB_ACCESS"
    INTERNSHIP_ACCESS = "INTERNSHIP_ACCESS"
    TRAINING_ENROLLMENT = "TRAINING_ENROLLMENT"
    CERTIFICATION = "CERTIFICATION"
    PRODUCT_OR_SERVICE = "PRODUCT_OR_SERVICE"
    APPLICATION_PROCESS = "APPLICATION_PROCESS"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class AnalysisConsistency(str, Enum):
    """How consistently the analysis signals agree."""
    HIGH = "HIGH"       # All signals within ~15 points
    MEDIUM = "MEDIUM"   # Spread < 35 points
    LOW = "LOW"         # Large disagreement — confidence reduced
    UNKNOWN = "UNKNOWN" # Not enough signals to compute


@dataclass
class WarningIndicator:
    type: str
    severity: str
    evidence: str
    description: str = ""
    title: str = ""                     # Short human-readable title
    source: str = "Rule Engine"         # Message Evidence | Rule Engine | Local ML | AI Contextual Analysis

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "evidence": self.evidence,
            "description": self.description,
            "title": self.title,
            "source": self.source,
        }


@dataclass
class AnalysisResult:
    """
    Normalized analysis result from any provider.
    This is the single internal schema used throughout the application.

    When analysis fails completely:
        risk_score = None
        risk_level = UNKNOWN
        verdict    = ANALYSIS_UNAVAILABLE
    """
    # Core risk assessment
    risk_score: Optional[float] = None     # 0–100, or None if analysis failed
    risk_level: str = RiskLevel.UNKNOWN
    verdict: str = Verdict.ANALYSIS_UNAVAILABLE
    confidence: float = 0.0               # 0.0–1.0
    warning_indicators: List[WarningIndicator] = field(default_factory=list)
    explanation: str = ""
    recommendation: str = ""
    evidence: List[str] = field(default_factory=list)

    # Opportunity classification (NEW)
    opportunity_type: str = OpportunityType.UNKNOWN
    payment_context: str = PaymentContext.NONE
    payment_required_for: str = PaymentRequiredFor.NONE

    # Score breakdown (optional, for display)
    ml_score: Optional[float] = None
    gemini_score: Optional[float] = None
    rule_score: Optional[float] = None

    # Analysis metadata
    analysis_consistency: str = AnalysisConsistency.UNKNOWN
    provider_used: str = "unknown"
    analysis_error: Optional[str] = None

    def is_unavailable(self) -> bool:
        """Return True if analysis failed and no risk assessment was made."""
        return (
            self.risk_score is None
            or self.verdict == Verdict.ANALYSIS_UNAVAILABLE
            or self.risk_level == RiskLevel.UNKNOWN
        )

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 1) if self.risk_score is not None else None,
            "risk_level": self.risk_level,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "warning_indicators": [w.to_dict() for w in self.warning_indicators],
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "opportunity_type": self.opportunity_type,
            "payment_context": self.payment_context,
            "payment_required_for": self.payment_required_for,
            "analysis_consistency": self.analysis_consistency,
            "ml_score": self.ml_score,
            "gemini_score": self.gemini_score,
            "rule_score": self.rule_score,
            "provider_used": self.provider_used,
            "analysis_error": self.analysis_error,
        }


# ── Risk level helpers ──────────────────────────────────────────────────────

def risk_level_from_score(score: float) -> str:
    """Derive risk level string from a 0–100 score (configurable thresholds)."""
    from backend.config import get_config
    try:
        cfg = get_config().get("risk_engine", {}).get("thresholds", {})
        t_critical = float(cfg.get("critical", 75))
        t_high = float(cfg.get("high", 50))
        t_medium = float(cfg.get("medium", 25))
    except Exception:
        t_critical, t_high, t_medium = 75, 50, 25

    if score >= t_critical:
        return RiskLevel.CRITICAL
    elif score >= t_high:
        return RiskLevel.HIGH
    elif score >= t_medium:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def verdict_from_score(score: float) -> str:
    """Derive verdict string from a 0–100 score."""
    if score >= 75:
        return Verdict.HIGH_RISK_SCAM
    elif score >= 50:
        return Verdict.POTENTIAL_SCAM
    elif score >= 25:
        return Verdict.SUSPICIOUS
    else:
        return Verdict.LIKELY_LEGIT


# ── Gemini response validation ──────────────────────────────────────────────

def _safe_enum(value: str, enum_cls, default):
    """Safely coerce a string into an enum value, returning default if invalid."""
    valid = {e.value for e in enum_cls}
    return value if value in valid else default


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
    risk_level = _safe_enum(risk_level, RiskLevel, risk_level_from_score(risk_score))

    verdict = data.get("verdict", verdict_from_score(risk_score))
    verdict = _safe_enum(verdict, Verdict, verdict_from_score(risk_score))

    # Opportunity classification fields
    opportunity_type = _safe_enum(
        data.get("opportunity_type", "UNKNOWN"), OpportunityType, OpportunityType.UNKNOWN
    )
    payment_context = _safe_enum(
        data.get("payment_context", "NONE"), PaymentContext, PaymentContext.NONE
    )
    payment_required_for = _safe_enum(
        data.get("payment_required_for", "NONE"), PaymentRequiredFor, PaymentRequiredFor.NONE
    )

    # Warning indicators
    raw_indicators = data.get("warning_indicators", [])
    indicators = []
    for raw in raw_indicators:
        if isinstance(raw, dict):
            indicators.append(WarningIndicator(
                type=raw.get("type", "UNKNOWN"),
                severity=raw.get("severity", "MEDIUM"),
                evidence=raw.get("evidence", ""),
                description=raw.get("description", ""),
                title=raw.get("title", raw.get("type", "").replace("_", " ").title()),
                source="AI Contextual Analysis",
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
        opportunity_type=opportunity_type,
        payment_context=payment_context,
        payment_required_for=payment_required_for,
        provider_used="gemini",
    )
