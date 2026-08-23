"""
Urgency and pressure tactics rule detector.
"""

import re
from typing import List
from backend.schemas import WarningIndicator

HIGH_PATTERNS = [
    (r"only\s*\d+\s*(?:seat|spot|position|slot)s?\s*(?:remaining|left|available)", "Limited seats claim"),
    (r"(?:offer|opportunity|position)\s*(?:expire|expires|closing|closes)\s*(?:today|tonight|now|soon)", "Expiring offer"),
    (r"act\s*now\b", "Act now pressure"),
    (r"(?:last|final)\s*chance", "Last chance urgency"),
    (r"today\s*only", "Today only offer"),
    (r"limited\s*(?:time|offer|seats|spots)", "Limited time/seats"),
    (r"(?:hurry|rush)\s*(?:up|now)?", "Hurry up pressure"),
    (r"(?:deadline|last\s*date)\s*(?:is\s*)?today", "Today's deadline"),
    (r"grab\s*(?:this\s*)?(?:opportunity|chance|offer)\s*(?:now|fast|quickly)", "Grab it now"),
]

MEDIUM_PATTERNS = [
    (r"\burgent\b", "Urgency keyword"),
    (r"\bimmediately\b", "Immediately keyword"),
    (r"\basap\b", "ASAP keyword"),
    (r"don't\s*(?:miss|delay|wait)", "Don't miss out"),
    (r"within\s*\d+\s*(?:hour|minute|day)s?", "Short timeframe given"),
    (r"(?:confirm|respond|reply)\s*(?:immediately|now|asap|quickly|fast)", "Immediate response demanded"),
    (r"first\s*come\s*first\s*serve", "First come first serve"),
    (r"exclusive\s*(?:offer|opportunity)", "Exclusive offer claim"),
    (r"seats?\s*(?:filling|filled)\s*(?:up|fast|quickly)", "Seats filling up"),
]


def detect_urgency_indicators(text: str) -> List[WarningIndicator]:
    """Detect urgency and pressure tactics in the input text."""
    lower_text = text.lower()
    indicators: List[WarningIndicator] = []

    for pattern, description in HIGH_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="URGENCY",
                severity="HIGH",
                evidence=evidence_text,
                description=description,
            ))

    for pattern, description in MEDIUM_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            indicators.append(WarningIndicator(
                type="PRESSURE_TACTICS",
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
