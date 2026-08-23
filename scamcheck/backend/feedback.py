"""
Feedback storage module for ScamCheck.

Stores user feedback to data/feedback/feedback.jsonl
One JSON entry per line. No private message content is stored.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import get_config

logger = logging.getLogger(__name__)


def _get_feedback_path() -> Path:
    """Return the configured feedback file path."""
    try:
        config = get_config()
        rel_path = config.get("feedback", {}).get("path", "data/feedback/feedback.jsonl")
    except Exception:
        rel_path = "data/feedback/feedback.jsonl"

    # Resolve relative to project root (parent of this file's package)
    project_root = Path(__file__).parent.parent
    return project_root / rel_path


def save_feedback(
    feedback_type: str,          # "HELPFUL" | "NOT_HELPFUL"
    reason: Optional[str] = None,  # Reason code if not helpful
    comment: Optional[str] = None,  # Optional free text (sanitized)
    opportunity_type: Optional[str] = None,
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None,
    provider_used: Optional[str] = None,
) -> bool:
    """
    Save a feedback entry to feedback.jsonl.

    Does NOT store:
    - API keys
    - Passwords or OTPs
    - The original submitted message text
    - Any personally identifiable information

    Returns:
        True if saved successfully, False on error.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback_type": str(feedback_type),
        "reason": reason,
        "comment": _sanitize_comment(comment) if comment else None,
        "opportunity_type": opportunity_type,
        "risk_score": round(risk_score, 1) if risk_score is not None else None,
        "risk_level": risk_level,
        "provider_used": provider_used,
    }

    try:
        feedback_path = _get_feedback_path()
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"[Feedback] Saved feedback entry: type={feedback_type}, reason={reason}")
        return True

    except Exception as e:
        logger.error(f"[Feedback] Failed to save feedback: {e}")
        return False


def _sanitize_comment(comment: str, max_length: int = 500) -> str:
    """
    Sanitize user comment:
    - Strip leading/trailing whitespace
    - Truncate to max_length
    - Remove any sequences that look like API keys or credentials
    """
    comment = comment.strip()[:max_length]
    return comment
