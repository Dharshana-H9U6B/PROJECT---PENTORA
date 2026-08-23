"""
Employment-related scam indicator detector.

Detects unrealistic salary claims, guaranteed selection,
suspicious contact methods, and fake company indicators.
"""

import re
from typing import List
from backend.schemas import WarningIndicator

HIGH_PATTERNS = [
    (r"no\s*interview\s*(?:required|needed)", "No interview required — highly suspicious"),
    (r"guaranteed\s*(?:job|employment|selection|placement)", "Guaranteed employment claim"),
    (r"100\s*%\s*(?:job|placement|selection)", "100% placement guarantee claim"),
    (r"selected\s*without\s*(?:interview|test|screening)", "Selection without screening"),
    (r"work\s*from\s*home.*(?:earn|₹|rs\.?)\s*\d+", "Unrealistic WFH earning claim"),
    (r"earn\s*(?:₹|rs\.?|\$)?\s*\d{4,}\s*(?:per\s*(?:day|hour))", "Unrealistic earnings per day/hour"),
    (r"(?:google|amazon|microsoft|apple|facebook|infosys|wipro|tcs)\s*(?:is\s*)?hiring", "Unverified big-tech hiring claim"),
    (r"congratulations.{0,50}(?:selected|chosen|hired)", "Congratulations selected — unsolicited offer"),
]

MEDIUM_PATTERNS = [
    (r"whatsapp\s*(?:only|number|contact|recruitment)", "WhatsApp-only recruitment"),
    (r"contact\s*(?:on|via|through|only)\s*whatsapp", "WhatsApp-only contact"),
    (r"telegram\s*(?:only|group|channel|contact)", "Telegram-only contact"),
    (r"gmail\.com.*(?:hr|recruitment|jobs|career)", "Corporate HR using Gmail"),
    (r"yahoo\.com.*(?:hr|recruitment|jobs|career)", "Corporate HR using Yahoo"),
    (r"hotmail\.com.*(?:hr|recruitment|jobs|career)", "Corporate HR using Hotmail"),
    (r"no\s*experience\s*(?:required|needed)", "No experience required claim"),
    (r"(?:fresher|freshers)\s*(?:welcome|preferred|required)", "Freshers welcome — verify legitimacy"),
    (r"part\s*time.*(?:₹|rs\.?)?\s*\d{4,}", "High part-time pay claim"),
    (r"work\s*from\s*home.*(?:guaranteed|assured)", "Guaranteed WFH offer"),
]


def detect_employment_indicators(text: str) -> List[WarningIndicator]:
    """Detect employment-related scam indicators."""
    lower_text = text.lower()
    indicators: List[WarningIndicator] = []

    for pattern, description in HIGH_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="UNVERIFIED_RECRUITMENT",
                severity="HIGH",
                evidence=evidence_text,
                description=description,
            ))

    for pattern, description in MEDIUM_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="SUSPICIOUS_CONTACT",
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
