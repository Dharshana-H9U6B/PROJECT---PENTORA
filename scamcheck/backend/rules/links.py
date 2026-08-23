"""
URL and link suspicious pattern detector.
"""

import re
from typing import List
from urllib.parse import urlparse

from backend.schemas import WarningIndicator

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "buff.ly", "short.link", "rb.gy", "cutt.ly", "is.gd",
    "tiny.cc", "lnkd.in", "youtu.be", "wp.me", "tr.im",
}

# Suspicious keywords in URLs
SUSPICIOUS_URL_KEYWORDS = [
    "login", "verify", "account", "secure", "update", "confirm",
    "click-here", "claim", "prize", "winner", "reward", "free",
    "job-apply", "apply-now", "hiring", "internship-offer",
]

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
)


def detect_link_indicators(text: str) -> List[WarningIndicator]:
    """Detect suspicious URLs and link patterns."""
    indicators: List[WarningIndicator] = []
    urls = URL_PATTERN.findall(text)

    if not urls:
        return indicators

    for url in urls:
        url_indicators = _analyze_url(url)
        indicators.extend(url_indicators)

    return indicators


def _analyze_url(url: str) -> List[WarningIndicator]:
    """Analyze a single URL for red flags."""
    indicators = []

    # Normalize: add scheme if missing
    if url.startswith("www."):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return indicators

    # Check for URL shorteners
    for shortener in URL_SHORTENERS:
        if shortener in domain:
            indicators.append(WarningIndicator(
                type="SUSPICIOUS_URL",
                severity="MEDIUM",
                evidence=url[:100],
                description=f"URL uses shortener service ({shortener}) which hides the actual destination.",
            ))
            break

    # Check for HTTP (non-HTTPS)
    if url.startswith("http://") and not url.startswith("http://localhost"):
        indicators.append(WarningIndicator(
            type="INSECURE_URL",
            severity="LOW",
            evidence=url[:100],
            description="URL uses HTTP instead of HTTPS — lacks encryption.",
        ))

    # Check for IP address instead of domain
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        indicators.append(WarningIndicator(
            type="SUSPICIOUS_URL",
            severity="HIGH",
            evidence=url[:100],
            description="URL uses an IP address instead of a domain name — highly suspicious.",
        ))

    # Check for excessive subdomains (possible impersonation)
    if domain and domain.count(".") >= 3:
        indicators.append(WarningIndicator(
            type="SUSPICIOUS_DOMAIN",
            severity="MEDIUM",
            evidence=url[:100],
            description=f"URL has excessive subdomains ({domain}) — possible impersonation.",
        ))

    # Check for suspicious keywords in URL
    url_lower = url.lower()
    for keyword in SUSPICIOUS_URL_KEYWORDS:
        if keyword in url_lower:
            indicators.append(WarningIndicator(
                type="SUSPICIOUS_URL",
                severity="MEDIUM",
                evidence=url[:100],
                description=f"URL contains suspicious keyword: '{keyword}'.",
            ))
            break

    return indicators
