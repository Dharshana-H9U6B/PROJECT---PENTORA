# -*- coding: utf-8 -*-
"""
ScamCheck -- Gemini Integration Diagnostic
Run from project root: python scratch/diag_gemini.py

Tests:
  1. .env loading and API key presence
  2. Gemini client initialization
  3. Model name resolution
  4. Text analysis (scam sentence)
  5. Image analysis (solid-color PIL image)
  6. Exception surfacing (no swallowing)
"""
import sys, os, traceback, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# --- Path setup -----------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

print("=" * 60)
print("STEP 1: .env loading")
print("=" * 60)
env_path = ROOT / ".env"
print(f"  .env path   : {env_path}")
print(f"  .env exists : {env_path.exists()}")
loaded = load_dotenv(env_path)
print(f"  dotenv loaded: {loaded}")

key = os.getenv("GEMINI_API_KEY")
model_env = os.getenv("GEMINI_MODEL")
print(f"  GEMINI_API_KEY present : {key is not None and len(key) > 0}")
print(f"  GEMINI_API_KEY length  : {len(key) if key else 0}")
print(f"  GEMINI_MODEL (env)     : {model_env!r}")

if not key:
    print("\n[FATAL] GEMINI_API_KEY not set — cannot proceed.")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Config resolution")
print("=" * 60)
from backend.config import get_config, get_gemini_api_key

cfg_key = get_gemini_api_key()
cfg = get_config()
model_from_cfg = cfg.get("gemini", {}).get("model")
print(f"  get_gemini_api_key() present : {cfg_key is not None and len(cfg_key) > 0}")
print(f"  Config model name            : {model_from_cfg!r}")

print()
print("=" * 60)
print("STEP 3: GeminiProvider initialization")
print("=" * 60)
from backend.models.gemini_provider import GeminiProvider

provider = GeminiProvider()
print(f"  is_available()  : {provider.is_available()}")
print(f"  provider_name() : {provider.provider_name()}")
print(f"  _init_error     : {provider._init_error!r}")
print(f"  _model_name     : {provider._model_name!r}")

if not provider.is_available():
    print(f"\n[FATAL] Provider not available: {provider._init_error}")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 4: Text analysis test")
print("=" * 60)
TEST_TEXT = (
    "Analyze this sentence for recruitment scam indicators: "
    "Pay Rs.2999 registration fee immediately to secure your internship."
)
print(f"  Input: {TEST_TEXT[:80]}...")
try:
    result = provider.analyze_text(TEST_TEXT)
    print(f"  risk_score   : {result.risk_score}")
    print(f"  risk_level   : {result.risk_level}")
    print(f"  verdict      : {result.verdict}")
    print(f"  confidence   : {result.confidence}")
    print(f"  indicators   : {len(result.warning_indicators)}")
    print(f"  explanation  : {result.explanation[:100]}...")
    print(f"  provider_used: {result.provider_used}")
    print("  [TEXT TEST PASSED]")
except Exception:
    print("  [TEXT TEST FAILED]")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 5: Image analysis test (synthetic PIL image)")
print("=" * 60)
from PIL import Image as PILImage

# Create a synthetic "screenshot-like" image with text content indicator
img = PILImage.new("RGB", (400, 200), color=(255, 255, 255))
try:
    img_fmt = img.format
    print(f"  PIL Image created: {img.size}, mode={img.mode}, format={img_fmt!r}")
    result_img = provider.analyze_image(img, context="Synthetic test image")
    print(f"  risk_score   : {result_img.risk_score}")
    print(f"  risk_level   : {result_img.risk_level}")
    print(f"  verdict      : {result_img.verdict}")
    print(f"  confidence   : {result_img.confidence}")
    print(f"  provider_used: {result_img.provider_used}")
    print(f"  explanation  : {result_img.explanation[:100]}...")
    print("  [IMAGE TEST PASSED]")
except Exception:
    print("  [IMAGE TEST FAILED]")
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 6: Fallback behaviour (bad model name)")
print("=" * 60)
from backend.models.gemini_provider import GeminiProvider as GP2

bad = GP2.__new__(GP2)
bad._client = provider._client
bad._model_name = "gemini-NONEXISTENT-model"
bad._initialized = True
bad._init_error = None

try:
    bad.analyze_text("test fallback")
    print("  [UNEXPECTED SUCCESS — should have raised]")
except Exception as e:
    print(f"  Exception type : {type(e).__name__}")
    print(f"  Exception msg  : {str(e)[:120]}")
    print("  [FALLBACK TEST PASSED — exception propagated correctly]")

print()
print("=" * 60)
print("ALL DIAGNOSTICS COMPLETE")
print("=" * 60)
