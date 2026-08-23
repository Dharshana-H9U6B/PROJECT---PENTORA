"""
URL Analyzer — lightweight heuristic URL analysis.

IMPORTANT: This analysis is heuristic only. It does not prove
that any website is malicious or safe. It identifies patterns
commonly associated with suspicious URLs.
"""

import re
from typing import List
from urllib.parse import urlparse

from backend.schemas import WarningIndicator

DISCLAIMER = (
    "URL analysis is heuristic and does not prove that a website is malicious. "
    "Always verify URLs independently before visiting."
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
)


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from the given text."""
    return URL_PATTERN.findall(text)


def analyze_urls(text: str) -> dict:
    """
    Analyze URLs found in text.

    Returns:
        {
            "urls_found": [...],
            "indicators": [...],
            "disclaimer": "...",
        }
    """
    urls = extract_urls(text)
    all_indicators: List[WarningIndicator] = []

    for url in urls:
        all_indicators.extend(_check_url(url))

    return {
        "urls_found": urls,
        "indicators": [ind.to_dict() for ind in all_indicators],
        "disclaimer": DISCLAIMER,
    }


def _check_url(url: str) -> List[WarningIndicator]:
    """Perform heuristic checks on a single URL."""
    indicators = []

    normalized = url if url.startswith("http") else "http://" + url

    try:
        parsed = urlparse(normalized)
        domain = parsed.netloc.lower().strip("www.")
    except Exception:
        return indicators

    # IP-based URL
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}", domain):
        indicators.append(WarningIndicator(
            type="SUSPICIOUS_URL",
            severity="HIGH",
            evidence=url[:120],
            description="URL uses an IP address instead of a domain name.",
        ))

    # HTTP only
    if normalized.startswith("http://"):
        indicators.append(WarningIndicator(
            type="INSECURE_URL",
            severity="LOW",
            evidence=url[:120],
            description="URL uses HTTP (not HTTPS). Data sent may not be encrypted.",
        ))

    # URL shorteners
    shorteners = ["bit.ly", "tinyurl.com", "t.co", "ow.ly", "goo.gl",
                  "short.link", "rb.gy", "cutt.ly", "is.gd"]
    for shortener in shorteners:
        if shortener in domain:
            indicators.append(WarningIndicator(
                type="SHORTENED_URL",
                severity="MEDIUM",
                evidence=url[:120],
                description=f"URL shortener ({shortener}) hides the true destination.",
            ))
            break

    # Excessive subdomains (impersonation)
    if domain.count(".") >= 3:
        indicators.append(WarningIndicator(
            type="SUSPICIOUS_DOMAIN",
            severity="MEDIUM",
            evidence=url[:120],
            description=f"URL has excessive subdomains, possible domain impersonation.",
        ))

    # Suspicious keywords in path
    path = parsed.path.lower()
    suspicious_keywords = ["login", "verify", "secure", "update", "confirm", "claim"]
    for kw in suspicious_keywords:
        if kw in path:
            indicators.append(WarningIndicator(
                type="SUSPICIOUS_URL",
                severity="MEDIUM",
                evidence=url[:120],
                description=f"URL path contains suspicious keyword: '{kw}'.",
            ))
            break

    return indicators
