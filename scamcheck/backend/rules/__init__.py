"""
Rules package — deterministic signal detection.
"""
from backend.rules.financial import detect_financial_indicators
from backend.rules.urgency import detect_urgency_indicators
from backend.rules.links import detect_link_indicators
from backend.rules.employment import detect_employment_indicators
from backend.rules.sensitive_data import detect_sensitive_data_indicators

__all__ = [
    "detect_financial_indicators",
    "detect_urgency_indicators",
    "detect_link_indicators",
    "detect_employment_indicators",
    "detect_sensitive_data_indicators",
]
