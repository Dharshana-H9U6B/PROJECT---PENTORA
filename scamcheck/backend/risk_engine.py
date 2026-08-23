"""
Risk Engine — combines signals from multiple sources into a final risk assessment.

Signal sources:
    1. Local ML classifier (probability-based)
    2. Gemini AI analysis
    3. Rule-based indicators

Scoring is CONTEXTUAL:
    - Payment context from Gemini is used to modulate the rule engine's contribution.
    - EMPLOYMENT_PAYMENT → full rule score contribution.
    - TRAINING_PAYMENT   → zero rule score contribution.
    - UNKNOWN context    → reduced contribution.

Weights are fully configurable via config.yaml.
"""

import logging
from typing import List, Optional, Dict, Tuple

from backend.config import get_config
from backend.schemas import (
    AnalysisResult,
    WarningIndicator,
    AnalysisConsistency,
    RiskLevel,
    Verdict,
    risk_level_from_score,
    verdict_from_score,
)
from backend.rules import (
    detect_financial_indicators,
    detect_urgency_indicators,
    detect_link_indicators,
    detect_employment_indicators,
    detect_sensitive_data_indicators,
)

logger = logging.getLogger(__name__)

# Severity weights for converting rule indicators to a score contribution
SEVERITY_WEIGHTS = {
    "CRITICAL": 35,
    "HIGH": 22,
    "MEDIUM": 10,
    "LOW": 4,
}


def _get_payment_context_multiplier(payment_context: Optional[str]) -> float:
    """
    Return the multiplier to apply to FINANCIAL rule scores based on payment context.
    This ensures that TRAINING_PAYMENT does not inflate the risk score.
    """
    try:
        config = get_config()
        multipliers = config.get("risk_engine", {}).get("payment_context_multipliers", {})
    except Exception:
        multipliers = {}

    defaults = {
        "EMPLOYMENT_PAYMENT": 1.0,
        "APPLICATION_PAYMENT": 0.6,
        "UNKNOWN": 0.4,
        "TRAINING_PAYMENT": 0.0,
        "PRODUCT_OR_SERVICE_PAYMENT": 0.0,
        "NONE": 0.2,
    }
    defaults.update(multipliers)
    return float(defaults.get(str(payment_context or "UNKNOWN"), 0.4))


def run_rule_engine(
    text: str,
    payment_context: Optional[str] = None,
) -> Tuple[List[WarningIndicator], float]:
    """
    Run all rule-based detectors on the input text.

    Args:
        text:            The input text.
        payment_context: PaymentContext string from Gemini (used to modulate financial rules).

    Returns:
        (indicators, rule_score) where rule_score is 0–100.
    """
    # Financial indicators — pass payment_context for context-aware scoring
    financial_indicators = detect_financial_indicators(text, payment_context=payment_context)
    other_indicators: List[WarningIndicator] = []
    other_indicators.extend(detect_urgency_indicators(text))
    other_indicators.extend(detect_link_indicators(text))
    other_indicators.extend(detect_employment_indicators(text))
    other_indicators.extend(detect_sensitive_data_indicators(text))

    all_indicators = financial_indicators + other_indicators

    # Score: financial indicators are multiplied by payment context multiplier
    payment_multiplier = _get_payment_context_multiplier(payment_context)

    financial_score = sum(
        SEVERITY_WEIGHTS.get(ind.severity, 5) for ind in financial_indicators
    ) * payment_multiplier

    other_score = sum(
        SEVERITY_WEIGHTS.get(ind.severity, 5) for ind in other_indicators
    )

    raw_score = financial_score + other_score
    rule_score = min(100.0, float(raw_score))

    logger.debug(
        f"[RuleEngine] payment_context={payment_context}, multiplier={payment_multiplier:.2f}, "
        f"financial_score={financial_score:.1f}, other_score={other_score:.1f}, total={rule_score:.1f}"
    )

    return all_indicators, rule_score


def calculate_final_risk(
    ml_score: Optional[float],
    gemini_score: Optional[float],
    rule_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate final risk score from component scores using configurable weights.

    Args:
        ml_score:     Local ML probability (0–100). None if unavailable.
        gemini_score: Gemini-derived score (0–100). None if unavailable.
        rule_score:   Rule-based score (0–100).
        weights:      Optional weight overrides. Reads from config by default.

    Returns:
        Final risk score (0–100).
    """
    if weights is None:
        config = get_config()
        re_cfg = config.get("risk_engine", {})
        weights = {
            "ml": float(re_cfg.get("ml_weight", 0.30)),
            "gemini": float(re_cfg.get("gemini_weight", 0.50)),
            "rules": float(re_cfg.get("rules_weight", 0.20)),
        }

    # Determine available components and redistribute weights
    available_weights: Dict[str, float] = {}
    if gemini_score is not None:
        available_weights["gemini"] = weights.get("gemini", 0.50)
    if ml_score is not None:
        available_weights["ml"] = weights.get("ml", 0.30)
    # Rule score is always available
    available_weights["rules"] = weights.get("rules", 0.20)

    # Normalize weights to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        return rule_score

    normalized = {k: v / total_weight for k, v in available_weights.items()}

    score = 0.0
    if "gemini" in normalized and gemini_score is not None:
        score += normalized["gemini"] * gemini_score
    if "ml" in normalized and ml_score is not None:
        score += normalized["ml"] * ml_score
    score += normalized["rules"] * rule_score

    return min(100.0, max(0.0, score))


def compute_analysis_consistency(
    gemini_score: Optional[float],
    ml_score: Optional[float],
    rule_score: Optional[float],
) -> Tuple[str, Optional[float]]:
    """
    Compute the consistency of analysis signals.

    Returns:
        (consistency: str, confidence_cap: Optional[float])
        confidence_cap is None if no cap should be applied.
    """
    try:
        config = get_config()
        cons_cfg = config.get("risk_engine", {}).get("consistency", {})
        high_threshold = float(cons_cfg.get("high_threshold", 15))
        medium_threshold = float(cons_cfg.get("medium_threshold", 35))
        low_cap = float(cons_cfg.get("low_confidence_cap", 0.65))
    except Exception:
        high_threshold, medium_threshold, low_cap = 15, 35, 0.65

    available = [s for s in [gemini_score, ml_score, rule_score] if s is not None]

    if len(available) < 2:
        return AnalysisConsistency.UNKNOWN, None

    spread = max(available) - min(available)

    if spread <= high_threshold:
        return AnalysisConsistency.HIGH, None
    elif spread <= medium_threshold:
        return AnalysisConsistency.MEDIUM, None
    else:
        return AnalysisConsistency.LOW, low_cap


def merge_indicators(
    rule_indicators: List[WarningIndicator],
    gemini_indicators: List[WarningIndicator],
) -> List[WarningIndicator]:
    """
    Merge rule and AI indicators, deduplicating where possible.
    Gemini indicators take precedence for the same type.
    """
    merged = list(gemini_indicators)
    existing_types = {ind.type for ind in merged}

    for ind in rule_indicators:
        if ind.type not in existing_types:
            merged.append(ind)
            existing_types.add(ind.type)

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    merged.sort(key=lambda x: severity_order.get(x.severity, 4))

    return merged


def build_final_result(
    text: str,
    gemini_result: Optional[AnalysisResult],
    ml_result: Optional[AnalysisResult],
) -> AnalysisResult:
    """
    Build the final combined AnalysisResult from all available signals.

    Args:
        text:          Original input text (for rule engine).
        gemini_result: Result from GeminiProvider (may be None).
        ml_result:     Result from LocalMLProvider (may be None).

    Returns:
        Final merged AnalysisResult.
    """
    # Extract payment_context from Gemini if available (used to modulate rules)
    payment_context = None
    if gemini_result:
        payment_context = getattr(gemini_result, "payment_context", None)

    # 1. Run rule engine — with payment context for contextual scoring
    rule_indicators, rule_score = run_rule_engine(text, payment_context=payment_context)

    # 2. Extract component scores
    gemini_score = gemini_result.risk_score if gemini_result else None
    ml_score = ml_result.risk_score if ml_result else None

    # 3. Calculate final score
    final_score = calculate_final_risk(ml_score, gemini_score, rule_score)

    # 4. Compute analysis consistency
    consistency, confidence_cap = compute_analysis_consistency(gemini_score, ml_score, rule_score)

    # 5. Merge indicators
    gemini_indicators = gemini_result.warning_indicators if gemini_result else []
    all_indicators = merge_indicators(rule_indicators, gemini_indicators)

    # 6. Determine provider info
    providers_used = []
    if gemini_result:
        providers_used.append("gemini")
    if ml_result:
        providers_used.append("local_ml")
    providers_used.append("rules")

    # 7. Prefer Gemini's explanation and recommendation if available
    if gemini_result:
        explanation = gemini_result.explanation
        recommendation = gemini_result.recommendation
        confidence = gemini_result.confidence

        # Add consistency note if signals disagree
        if consistency == AnalysisConsistency.LOW:
            explanation += (
                " Note: The available analysis signals are not fully consistent. "
                "Verify this opportunity independently before drawing conclusions."
            )

        # Cap confidence if consistency is LOW
        if confidence_cap is not None:
            confidence = min(confidence, confidence_cap)
    else:
        # Fallback explanation from rule score
        indicator_count = len(all_indicators)
        if indicator_count == 0:
            explanation = "No significant scam indicators were detected in the provided opportunity."
            recommendation = "This opportunity appears relatively low risk. However, always verify independently."
        else:
            high_count = sum(1 for i in all_indicators if i.severity in ("HIGH", "CRITICAL"))
            explanation = (
                f"The rule-based analysis detected {indicator_count} warning indicator(s), "
                f"including {high_count} high/critical severity flag(s). "
                "Gemini AI analysis was unavailable for this assessment."
            )
            recommendation = (
                "Exercise caution. Verify the opportunity through the company's official website "
                "before taking any action or making any payment."
            )
        confidence = min(0.9, rule_score / 100.0 + 0.2) if rule_score > 0 else 0.3

    # 8. Extract opportunity classification from Gemini
    opportunity_type = getattr(gemini_result, "opportunity_type", "UNKNOWN") if gemini_result else "UNKNOWN"
    payment_context_val = getattr(gemini_result, "payment_context", "NONE") if gemini_result else "NONE"
    payment_required_for = getattr(gemini_result, "payment_required_for", "NONE") if gemini_result else "NONE"

    # 9. Assemble final result
    result = AnalysisResult(
        risk_score=final_score,
        risk_level=risk_level_from_score(final_score),
        verdict=verdict_from_score(final_score),
        confidence=confidence,
        warning_indicators=all_indicators,
        explanation=explanation,
        recommendation=recommendation,
        ml_score=ml_score,
        gemini_score=gemini_score,
        rule_score=rule_score,
        provider_used="+".join(providers_used),
        opportunity_type=opportunity_type,
        payment_context=payment_context_val,
        payment_required_for=payment_required_for,
        analysis_consistency=consistency,
    )

    return result
