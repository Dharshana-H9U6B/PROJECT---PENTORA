"""
Configuration loader for ScamCheck.
Reads config.yaml and merges with environment variables.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict:
    """
    Load and return the application configuration.

    Priority (highest first):
        1. Environment variables
        2. config.yaml
    """
    config = _load_yaml(_ROOT / "config.yaml")

    # Inject environment variables for secrets/overrides
    gemini_cfg = config.setdefault("gemini", {})
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        gemini_cfg["model"] = env_model

    # Make sure model has a sensible default
    gemini_cfg.setdefault("model", "gemini-2.0-flash")

    return config


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY")


# Singleton config loaded once
_CONFIG: dict | None = None


def get_config() -> dict:
    """Return the singleton application config."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG
