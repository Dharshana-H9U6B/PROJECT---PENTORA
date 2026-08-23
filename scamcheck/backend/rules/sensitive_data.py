"""
Sensitive data request detector.

Detects requests for OTPs, passwords, banking credentials,
and personal identification information.
"""

import re
from typing import List
from backend.schemas import WarningIndicator

CRITICAL_PATTERNS = [
    (r"\botp\b", "OTP request — legitimate employers never ask for OTPs"),
    (r"one.?time.?password", "One-time password request"),
    (r"upi\s*pin", "UPI PIN request"),
    (r"atm\s*pin", "ATM PIN request"),
    (r"cvv\b", "CVV request"),
    (r"card\s*(?:number|details|pin)", "Credit/debit card details request"),
    (r"(?:net|internet)\s*banking\s*(?:id|password|credentials)", "Net banking credentials request"),
    (r"bank\s*account\s*(?:number|details|password)", "Bank account details request"),
    (r"share\s*(?:your\s*)?password", "Password sharing request"),
]

HIGH_PATTERNS = [
    (r"aadhaar\s*(?:number|card|details)", "Aadhaar number request"),
    (r"pan\s*(?:number|card|details)\s*(?:for|to|required|mandatory)", "PAN card number request"),
    (r"passport\s*(?:number|copy|details)\s*(?:for|to|required|scan)", "Passport details request"),
    (r"enter\s*(?:your\s*)?(?:password|pin|otp|cvv)", "Credential entry request"),
]

MEDIUM_PATTERNS = [
    (r"date\s*of\s*birth.*(?:for|to\s*verify)", "Date of birth for verification"),
    (r"mother.?s\s*maiden\s*name", "Mother's maiden name request"),
    (r"home\s*address.*(?:for|to\s*(?:send|deliver|ship))", "Home address for delivery"),
]


def detect_sensitive_data_indicators(text: str) -> List[WarningIndicator]:
    """Detect requests for sensitive personal or financial data."""
    lower_text = text.lower()
    indicators: List[WarningIndicator] = []

    for pattern, description in CRITICAL_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="SENSITIVE_DATA_REQUEST",
                severity="CRITICAL",
                evidence=evidence_text,
                description=description,
            ))

    for pattern, description in HIGH_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="PERSONAL_INFO_REQUEST",
                severity="HIGH",
                evidence=evidence_text,
                description=description,
            ))

    for pattern, description in MEDIUM_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="PERSONAL_INFO_REQUEST",
                severity="MEDIUM",
                evidence=evidence_text,
                description=description,
            ))

    return indicators


def _extract_context(text: str, start: int, end: int, window: int = 60) -> str:
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].strip()
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."
    return snippet
