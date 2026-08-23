"""
Risk Engine — combines signals from multiple sources into a final risk assessment.

Signal sources:
    1. Local ML classifier (probability-based)
    2. Gemini AI analysis
    3. Rule-based indicators

Weights are fully configurable via config.yaml.
"""

import logging
from typing import List, Optional, Dict

from backend.config import get_config
from backend.schemas import (
    AnalysisResult,
    WarningIndicator,
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

# Severity weights for converting rule indicators to a score
SEVERITY_WEIGHTS = {
    "CRITICAL": 30,
    "HIGH": 20,
    "MEDIUM": 10,
    "LOW": 5,
}


def run_rule_engine(text: str) -> tuple[List[WarningIndicator], float]:
    """
    Run all rule-based detectors on the input text.

    Returns:
        (indicators, rule_score) where rule_score is 0–100.
    """
    all_indicators: List[WarningIndicator] = []
    all_indicators.extend(detect_financial_indicators(text))
    all_indicators.extend(detect_urgency_indicators(text))
    all_indicators.extend(detect_link_indicators(text))
    all_indicators.extend(detect_employment_indicators(text))
    all_indicators.extend(detect_sensitive_data_indicators(text))

    # Cap raw score at 100
    raw_score = sum(SEVERITY_WEIGHTS.get(ind.severity, 5) for ind in all_indicators)
    rule_score = min(100.0, float(raw_score))

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
        return rule_score  # fallback

    normalized = {k: v / total_weight for k, v in available_weights.items()}

    score = 0.0
    if "gemini" in normalized and gemini_score is not None:
        score += normalized["gemini"] * gemini_score
    if "ml" in normalized and ml_score is not None:
        score += normalized["ml"] * ml_score
    score += normalized["rules"] * rule_score

    return min(100.0, max(0.0, score))


def merge_indicators(
    rule_indicators: List[WarningIndicator],
    gemini_indicators: List[WarningIndicator],
) -> List[WarningIndicator]:
    """
    Merge rule and AI indicators, deduplicating where possible.
    Gemini indicators take precedence for the same type.
    """
    # Start with Gemini's indicators (richer descriptions)
    merged = list(gemini_indicators)
    existing_types = {ind.type for ind in merged}

    # Add rule indicators that are not already represented
    for ind in rule_indicators:
        if ind.type not in existing_types:
            merged.append(ind)
            existing_types.add(ind.type)
        else:
            # Add as supplementary evidence if it adds distinct detail
            pass

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
    # 1. Run rule engine
    rule_indicators, rule_score = run_rule_engine(text)

    # 2. Extract component scores
    gemini_score = gemini_result.risk_score if gemini_result else None
    ml_score = ml_result.risk_score if ml_result else None

    # 3. Calculate final score
    final_score = calculate_final_risk(ml_score, gemini_score, rule_score)

    # 4. Merge indicators
    gemini_indicators = gemini_result.warning_indicators if gemini_result else []
    all_indicators = merge_indicators(rule_indicators, gemini_indicators)

    # 5. Determine provider info
    providers_used = []
    if gemini_result:
        providers_used.append("gemini")
    if ml_result:
        providers_used.append("local_ml")
    providers_used.append("rules")

    # 6. Prefer Gemini's explanation and recommendation if available
    if gemini_result:
        explanation = gemini_result.explanation
        recommendation = gemini_result.recommendation
        confidence = gemini_result.confidence
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
                "before taking any action."
            )
        confidence = min(0.9, rule_score / 100.0 + 0.2) if rule_score > 0 else 0.3

    # 7. Assemble final result
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
    )

    return result
