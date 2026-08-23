"""
Financial red-flag rule detector — context-aware.

Payment detection is CONTEXTUAL:
  - Payment to obtain employment → HIGH risk signal
  - Payment for training/course/product → LOW or zero risk signal
  - Unknown payment context → MEDIUM risk signal

The payment_context parameter (from Gemini analysis) is used to
suppress or modulate rule scores when context is already known.
"""

import re
from typing import List, Optional
from backend.schemas import WarningIndicator

# ── Employment payment patterns (high risk) ────────────────────────────────
# These specifically link payment to obtaining a job/internship.
EMPLOYMENT_PAYMENT_PATTERNS = [
    (r"pay\s*(?:₹|rs\.?|inr)?\s*\d[\d,]*\s*(?:to\s*)?(?:confirm|secure|get|join|start)\s+(?:your\s+)?(?:internship|job|position|offer|placement|appointment)",
     "Payment to confirm employment"),
    (r"(?:registration|processing|joining|onboarding)\s*fee\s*(?:to|for|before)\s+(?:joining|starting|getting|confirming|securing)",
     "Fee required before joining"),
    (r"pay\s*(?:₹|rs\.?|inr)?\s*\d[\d,]*\s*(?:registration|processing|security|deposit)\s*fee",
     "Upfront fee request"),
    (r"security\s*deposit\s*(?:for|to)\s*(?:confirm|secure|get|join)",
     "Security deposit for employment"),
    (r"refundable\s*(?:deposit|amount)\s*(?:to|for)\s*(?:confirm|secure|join)",
     "Refundable deposit (common scam tactic)"),
    (r"advance\s*(?:fee|payment|amount)\s*(?:for|to)\s*(?:job|internship|offer|placement)",
     "Advance fee for employment"),
    (r"wire\s*transfer\s*(?:to|for)\s*(?:confirm|secure|join|start)",
     "Wire transfer to confirm employment"),
    (r"send\s*(?:₹|rs\.?|inr|money|amount|cash)\s*(?:to|for)\s*(?:confirm|secure|get|join)",
     "Request to send money to confirm position"),
    (r"pay\s*(?:before|first|now|immediately|today)\s*(?:to|and)\s*(?:join|start|get|confirm|secure)",
     "Pay immediately to start/join"),
    (r"training\s*(?:fee|cost|charges)\s*(?:must\s*be\s*paid|required|is\s*mandatory)\s*(?:before|to)\s*(?:join|start|begin\s*work)",
     "Training fee required before work begins"),
]

# ── Generic upfront payment (medium risk when context unknown) ─────────────
GENERIC_PAYMENT_PATTERNS = [
    (r"registration\s*fee", "Registration fee mentioned"),
    (r"processing\s*fee", "Processing fee mentioned"),
    (r"security\s*deposit", "Security deposit mentioned"),
    (r"pay\s*(?:to\s*confirm|before\s*joining|first\s*then)", "Pay-first phrasing"),
    (r"advance\s*(?:fee|payment|amount)", "Advance payment mentioned"),
    (r"refundable\s*deposit", "Refundable deposit claim"),
    (r"wire\s*transfer", "Wire transfer mentioned"),
    (r"send\s*(?:₹|rs\.?|inr|money|amount|cash)", "Request to send money"),
]

# ── Training/course payment (low risk, legitimate context) ─────────────────
TRAINING_PAYMENT_PATTERNS = [
    r"enro(?:l|ll)\s*(?:at|for|in)\s*(?:just\s+)?(?:₹|rs\.?|inr)?\s*[\d,]+",
    r"course\s*fee\s*(?:is\s*|:?\s*)(?:₹|rs\.?|inr)?\s*[\d,]+",
    r"certification\s*fee\s*(?:is\s*|:?\s*)(?:₹|rs\.?|inr)?\s*[\d,]+",
    r"program\s*fee\s*(?:is\s*|:?\s*)(?:₹|rs\.?|inr)?\s*[\d,]+",
    r"(?:training|workshop|bootcamp|seminar)\s*fee",
    r"(?:₹|rs\.?)\s*[\d,]+\s*\+\s*gst",
]

# ── Negation phrases that indicate a legitimate "no fee" context ───────────
_NEGATION_PHRASES = [
    r"no\s+(?:registration|processing|joining|onboarding|advance|upfront)\s+fee",
    r"no\s+(?:fee|fees|charge|charges|payment|payments)\s+(?:required|charged|asked|collected)",
    r"(?:free\s+of\s+cost|zero\s+fee|no\s+cost|no\s+charge)",
    r"(?:will\s+not|won.t|does\s+not|don.t)\s+(?:ask|charge|request|collect)\s+(?:any\s+)?(?:fee|payment|money|amount)",
    r"no\s+advance\s+payment\s+required",
    r"absolutely\s+free",
    r"at\s+no\s+(?:cost|charge)",
]

# ── UPI/banking patterns (medium risk on their own) ─────────────────────────
BANKING_PATTERNS = [
    (r"upi\s*(?:id|payment|transfer|link)", "UPI payment request"),
    (r"paytm|phonepe|gpay|google\s*pay", "Mobile payment app mentioned"),
    (r"bank\s*(?:transfer|deposit)\s*(?:to|of)", "Bank transfer request"),
    (r"account\s*(?:number|no\.?)\s*(?:to|for)\s*(?:pay|transfer|deposit)", "Bank account number requested for payment"),
]


def detect_financial_indicators(
    text: str,
    payment_context: Optional[str] = None,
) -> List[WarningIndicator]:
    """
    Detect financial red flags in the input text.

    Args:
        text:            The opportunity message/description.
        payment_context: PaymentContext string from Gemini analysis, if available.

    Returns:
        List of WarningIndicator objects for each detected pattern.
    """
    lower_text = text.lower()
    indicators: List[WarningIndicator] = []
    seen_types: set = set()

    # 1. Check if payment context is already known to be benign
    benign_contexts = {
        "TRAINING_PAYMENT",
        "PRODUCT_OR_SERVICE_PAYMENT",
    }
    context_is_benign = payment_context in benign_contexts
    context_is_employment = payment_context == "EMPLOYMENT_PAYMENT"

    # 2. Check for global negation (e.g. "no registration fee required")
    has_negation = _has_global_negation(lower_text)

    # 3. Detect explicit employment payment patterns → always HIGH (negation doesn't apply here)
    for pattern, description in EMPLOYMENT_PAYMENT_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            evidence_text = _extract_context(text, match.start(), match.end())
            ind_type = "EMPLOYMENT_PAYMENT"
            if ind_type not in seen_types:
                indicators.append(WarningIndicator(
                    type=ind_type,
                    severity="HIGH",
                    title="Payment required for employment",
                    evidence=evidence_text,
                    description=description,
                    source="Message Evidence",
                ))
                seen_types.add(ind_type)
            break

    # 4. Check for training-payment context (suppress generic rules)
    is_training_context = _is_training_context(lower_text)

    # 5. Generic payment patterns — skip if benign/negated/training/already flagged
    if (
        not context_is_benign
        and not is_training_context
        and not has_negation
        and "EMPLOYMENT_PAYMENT" not in seen_types
    ):
        for pattern, description in GENERIC_PAYMENT_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                if _is_salary_context(lower_text, match.start(), match.end()):
                    continue

                evidence_text = _extract_context(text, match.start(), match.end())

                if context_is_employment:
                    severity = "HIGH"
                    title = "Payment required for employment"
                    src = "Message Evidence"
                elif payment_context == "UNKNOWN" or payment_context is None:
                    severity = "MEDIUM"
                    title = "Upfront payment request"
                    src = "Rule Engine"
                else:
                    severity = "MEDIUM"
                    title = "Fee or payment request"
                    src = "Rule Engine"

                ind_type = "UPFRONT_PAYMENT"
                if ind_type not in seen_types:
                    indicators.append(WarningIndicator(
                        type=ind_type,
                        severity=severity,
                        title=title,
                        evidence=evidence_text,
                        description=description,
                        source=src,
                    ))
                    seen_types.add(ind_type)
                break

    # 6. UPI/banking patterns — medium severity, always relevant (unless negation)
    if not has_negation:
        for pattern, description in BANKING_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                evidence_text = _extract_context(text, match.start(), match.end())
                indicators.append(WarningIndicator(
                    type="SUSPICIOUS_PAYMENT_METHOD",
                    severity="MEDIUM",
                    title="Suspicious payment method",
                    evidence=evidence_text,
                    description=description,
                    source="Rule Engine",
                ))
                break

    return indicators


def _extract_context(text: str, start: int, end: int, window: int = 70) -> str:
    """Extract a snippet of text around a match for use as evidence."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].strip()
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."
    return snippet


def _has_global_negation(lower_text: str) -> bool:
    """Return True if the text explicitly states there is no fee/payment required."""
    for pattern in _NEGATION_PHRASES:
        if re.search(pattern, lower_text):
            return True
    return False


# Words that indicate a salary/income context rather than a fee request
_SALARY_CONTEXT_WORDS = [
    "stipend", "salary", "earn ", "earns", "package",
    "per month", "/month", "per annum", "lpa", "per year",
    "annually", "compensation", "income", "remuneration",
    "you will receive", "you will earn", "you will be paid",
]


def _is_salary_context(lower_text: str, match_start: int, match_end: int, window: int = 80) -> bool:
    """Return True if the match appears in a salary/income context."""
    ctx_start = max(0, match_start - window)
    ctx_end = min(len(lower_text), match_end + window)
    context = lower_text[ctx_start:ctx_end]
    return any(word in context for word in _SALARY_CONTEXT_WORDS)


def _is_training_context(lower_text: str) -> bool:
    """Return True if the text clearly describes a training/course purchase context."""
    for pattern in TRAINING_PAYMENT_PATTERNS:
        if re.search(pattern, lower_text):
            return True
    return False
