"""
Financial red-flag rule detector.

Detects requests for upfront payments, fees, and financial transfers.
"""

import re
from typing import List
from backend.schemas import WarningIndicator

# High-severity financial patterns
HIGH_PATTERNS = [
    (r"registration\s*fee", "Registration fee request"),
    (r"processing\s*fee", "Processing fee request"),
    (r"security\s*deposit", "Security deposit request"),
    (r"training\s*fee", "Training fee request"),
    (r"pay\s*(?:₹|rs\.?|inr|rupee)", "Payment request in Indian currency"),
    (r"₹\s*\d+", "Specific amount in Indian rupees"),
    (r"send\s*(?:₹|rs\.?|inr|money|amount|cash)", "Request to send money"),
    (r"deposit\s*(?:₹|rs\.?|inr|amount|money)", "Deposit money request"),
    (r"pay\s*(?:to\s*confirm|before|first|now|immediately|today)", "Pay to confirm position"),
    (r"wire\s*transfer", "Wire transfer request"),
    (r"advance\s*(?:fee|payment|amount)", "Advance fee request"),
    (r"refundable\s*deposit", "Refundable deposit (common scam tactic)"),
]

# Medium-severity financial patterns
MEDIUM_PATTERNS = [
    (r"payment\s*required", "Payment requirement mentioned"),
    (r"fee\s*(?:must\s*be|is|will\s*be)\s*paid", "Fee must be paid"),
    (r"upi\s*(?:id|payment|transfer)", "UPI payment request"),
    (r"paytm|phonepe|gpay|google\s*pay", "Mobile payment app mentioned in suspicious context"),
    (r"bank\s*(?:transfer|deposit)", "Bank transfer mentioned"),
    (r"amount\s*(?:to\s*be\s*paid|payable)", "Amount payable mentioned"),
    (r"fee\s*(?:of|is|:)\s*(?:₹|rs\.?|\$|€)?[\s]*\d+", "Specific fee amount mentioned"),
]


def detect_financial_indicators(text: str) -> List[WarningIndicator]:
    """
    Detect financial red flags in the input text.

    Args:
        text: The opportunity message/description.

    Returns:
        List of WarningIndicator objects for each detected pattern.
    """
    lower_text = text.lower()
    indicators: List[WarningIndicator] = []
    seen_types: set = set()

    for pattern, description in HIGH_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicator_type = "UPFRONT_PAYMENT"
            if indicator_type not in seen_types:
                indicators.append(WarningIndicator(
                    type=indicator_type,
                    severity="HIGH",
                    evidence=evidence_text,
                    description=description,
                ))
                seen_types.add(indicator_type)
            # Allow additional HIGH entries for distinct matches
            else:
                indicators.append(WarningIndicator(
                    type="FINANCIAL_REQUEST",
                    severity="HIGH",
                    evidence=evidence_text,
                    description=description,
                ))

    for pattern, description in MEDIUM_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="FINANCIAL_RED_FLAG",
                severity="MEDIUM",
                evidence=evidence_text,
                description=description,
            ))

    return indicators


def _extract_context(text: str, start: int, end: int, window: int = 60) -> str:
    """Extract a snippet of text around a match for use as evidence."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].strip()
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."
    return snippet
