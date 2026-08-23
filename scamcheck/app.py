"""
ScamCheck — AI-Powered Internship & Job Opportunity Verification
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# ── Setup ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

# Page config must be first Streamlit call
st.set_page_config(
    page_title="ScamCheck — Opportunity Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ScamCheck — AI-Powered Internship & Job Opportunity Verification. Built for Hackspora 2.0.",
    },
)

# ── Imports (after path setup) ─────────────────────────────────────────────────
from backend.services.analysis_service import get_analysis_service
from backend.schemas import AnalysisResult, RiskLevel, Verdict
from backend.input_processors.text_processor import validate_text_input, build_structured_text
from backend.feedback import save_feedback

MAX_INPUT_CHARS = 5000


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Light corporate background ── */
    .stApp {
        background-color: #f1f5f9;
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #334155 !important;
        font-size: 0.9rem !important;
    }

    /* ── Cards ── */
    .sc-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .sc-card-tight {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* ── Header ── */
    .sc-header {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 1.25rem 0 1rem;
        margin-bottom: 1.5rem;
    }
    .sc-brand {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .sc-brand-icon {
        color: #1d4ed8;
        margin-right: 0.4rem;
    }
    .sc-subtitle {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 400;
        margin-top: 0.1rem;
    }
    .sc-status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #16a34a;
    }
    .sc-status-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #dc2626;
    }
    .sc-status-dot-green { width: 7px; height: 7px; background: #16a34a; border-radius: 50%; display: inline-block; }
    .sc-status-dot-red { width: 7px; height: 7px; background: #dc2626; border-radius: 50%; display: inline-block; }

    /* ── Hero ── */
    .sc-hero {
        padding: 2rem 0 1.5rem;
        text-align: center;
    }
    .sc-hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .sc-hero-sub {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.75rem;
        line-height: 1.6;
    }
    .sc-feature-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.25rem;
        flex-wrap: wrap;
    }
    .sc-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-size: 0.8rem;
        color: #475569;
        font-weight: 500;
    }

    /* ── Risk Score Display ── */
    .sc-score-card {
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
        border: 1.5px solid;
    }
    .sc-score-card-low    { background: #f0fdf4; border-color: #86efac; }
    .sc-score-card-medium { background: #fffbeb; border-color: #fcd34d; }
    .sc-score-card-high   { background: #fff7ed; border-color: #fdba74; }
    .sc-score-card-critical { background: #fef2f2; border-color: #fca5a5; }
    .sc-score-card-unknown { background: #f8fafc; border-color: #cbd5e1; }

    .sc-score-num {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.04em;
    }
    .sc-score-denom { font-size: 1.5rem; font-weight: 400; color: #94a3b8; }
    .sc-score-label {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-top: 0.4rem;
        text-transform: uppercase;
    }
    .sc-score-verdict {
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.35rem;
    }

    /* Semantic text colors */
    .c-low      { color: #16a34a; }
    .c-medium   { color: #d97706; }
    .c-high     { color: #ea580c; }
    .c-critical { color: #dc2626; }
    .c-unknown  { color: #64748b; }

    /* ── Risk Bar ── */
    .sc-risk-bar-wrap {
        background: #f1f5f9;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 0.75rem 0 0.25rem;
    }
    .sc-risk-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.4s ease;
    }

    /* ── Meta info cards (opportunity type, payment context, etc.) ── */
    .sc-meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .sc-meta-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .sc-meta-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .sc-meta-value {
        font-size: 0.875rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* ── Warning Indicator Cards ── */
    .sc-indicator {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .sc-ind-critical { border-left-color: #dc2626; background: #fff5f5; }
    .sc-ind-high     { border-left-color: #ea580c; background: #fff8f5; }
    .sc-ind-medium   { border-left-color: #d97706; background: #fffdf5; }
    .sc-ind-low      { border-left-color: #16a34a; background: #f9fffe; }

    .sc-ind-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .sc-ind-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.9rem;
    }
    .sc-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .sc-badge-critical { background: #fee2e2; color: #991b1b; }
    .sc-badge-high     { background: #ffedd5; color: #c2410c; }
    .sc-badge-medium   { background: #fef3c7; color: #b45309; }
    .sc-badge-low      { background: #dcfce7; color: #15803d; }

    .sc-source-tag {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-left: auto;
        font-weight: 500;
    }
    .sc-ind-desc {
        color: #475569;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }
    .sc-evidence-box {
        background: #f8fafc;
        border-left: 3px solid #cbd5e1;
        border-radius: 4px;
        padding: 0.4rem 0.75rem;
        font-size: 0.82rem;
        color: #334155;
        font-style: italic;
        margin-top: 0.3rem;
    }

    /* ── Section headers ── */
    .sc-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* ── Analysis unavailable ── */
    .sc-unavailable {
        background: #fffbeb;
        border: 1.5px dashed #fcd34d;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .sc-unavail-icon { font-size: 2.5rem; }
    .sc-unavail-title { font-size: 1.25rem; font-weight: 700; color: #92400e; margin-top: 0.5rem; }
    .sc-unavail-body { color: #78350f; font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.6; }

    /* ── Disclaimer / bias note ── */
    .sc-disclaimer {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.5;
        margin-top: 1.25rem;
    }

    /* ── Primary button ── */
    .stButton > button {
        background: #1d4ed8 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: background 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(29,78,216,0.2) !important;
    }
    .stButton > button:hover {
        background: #1e40af !important;
    }

    /* ── Demo sample quick buttons ── */
    .sc-sample-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.25rem;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        color: #64748b !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.45rem 1.25rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07) !important;
    }

    /* ── Text inputs ── */
    .stTextArea textarea, .stTextInput input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    }

    /* ── Feedback section ── */
    .sc-feedback-wrap {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-top: 1.25rem;
    }

    /* ── How it works steps ── */
    .sc-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .sc-step-num {
        width: 36px;
        height: 36px;
        background: #1d4ed8;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .sc-step-content { flex: 1; }
    .sc-step-title { font-weight: 600; color: #0f172a; font-size: 0.95rem; }
    .sc-step-desc { color: #64748b; font-size: 0.875rem; margin-top: 0.15rem; line-height: 1.5; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Char counter */
    .sc-char-count {
        font-size: 0.75rem;
        color: #94a3b8;
        text-align: right;
        margin-top: 0.25rem;
    }
    .sc-char-warn { color: #ea580c; }

    /* Separator */
    hr { border-color: #e2e8f0; }

    </style>
    """, unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def _risk_css(risk_level: str) -> str:
    m = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low", "UNKNOWN": "unknown"}
    return m.get(risk_level, "unknown")


def _severity_css(severity: str) -> str:
    return {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(severity, "medium")


def _severity_icon(severity: str) -> str:
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")


def _verdict_label(verdict: str) -> str:
    labels = {
        "LIKELY_LEGIT": "Likely Legitimate",
        "SUSPICIOUS": "Suspicious",
        "POTENTIAL_SCAM": "Potential Scam",
        "HIGH_RISK_SCAM": "High Risk Scam",
        "ANALYSIS_UNAVAILABLE": "Analysis Unavailable",
    }
    return labels.get(verdict, verdict)


def _opp_type_label(val: str) -> str:
    labels = {
        "JOB": "Job Opportunity", "INTERNSHIP": "Internship",
        "PAID_TRAINING": "Paid Training", "CERTIFICATION": "Certification",
        "SCHOLARSHIP": "Scholarship", "EVENT": "Event",
        "OTHER": "Other", "UNKNOWN": "Unknown",
    }
    return labels.get(val, val.replace("_", " ").title())


def _payment_ctx_label(val: str) -> str:
    labels = {
        "EMPLOYMENT_PAYMENT": "Employment Payment",
        "TRAINING_PAYMENT": "Training Payment",
        "PRODUCT_OR_SERVICE_PAYMENT": "Product / Service",
        "APPLICATION_PAYMENT": "Application Fee",
        "UNKNOWN": "Unknown", "NONE": "None",
    }
    return labels.get(val, val.replace("_", " ").title())


def _consistency_label(val: str) -> str:
    labels = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "UNKNOWN": "N/A"}
    return labels.get(val, val)


def _bar_color(risk_level: str) -> str:
    return {
        "LOW": "#16a34a", "MEDIUM": "#d97706",
        "HIGH": "#ea580c", "CRITICAL": "#dc2626", "UNKNOWN": "#94a3b8",
    }.get(risk_level, "#94a3b8")


def load_demo_examples():
    demo_dir = ROOT / "data" / "demo"
    scam_examples, legit_examples = [], []
    try:
        with open(demo_dir / "scam_examples.json", "r", encoding="utf-8") as f:
            scam_examples = json.load(f)
    except Exception:
        pass
    try:
        with open(demo_dir / "legitimate_examples.json", "r", encoding="utf-8") as f:
            legit_examples = json.load(f)
    except Exception:
        pass
    return scam_examples, legit_examples


# ── Result rendering ──────────────────────────────────────────────────────────

def render_analysis_unavailable(result: AnalysisResult):
    """Render a clear 'analysis unavailable' state — never shows 0/100 LOW."""
    diag_err = result.analysis_error or "The AI analysis service is currently unavailable."
    st.markdown(f"""
    <div class="sc-unavailable">
        <div class="sc-unavail-icon">⚠️</div>
        <div class="sc-unavail-title">Analysis Unavailable</div>
        <div class="sc-unavail-body">
            {result.explanation or diag_err}<br><br>
            <strong>Risk Score: N/A</strong> — No risk classification has been made.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **What you can do:**
    - **Configure your API key** — add `GEMINI_API_KEY` to the `.env` file and restart.
    - **Use the Message tab** — paste the opportunity text to analyze it as text.
    - **Retry** — transient network issues may resolve on retry.
    """)


def render_result(result: AnalysisResult):
    """Render the full analysis result UI."""
    # ── Handle unavailable state ──
    if result.is_unavailable():
        render_analysis_unavailable(result)
        return

    risk_cls = _risk_css(result.risk_level)
    score = int(result.risk_score)
    bar_pct = score
    bar_color = _bar_color(result.risk_level)

    # ── Score card ──
    st.markdown(f"""
    <div class="sc-score-card sc-score-card-{risk_cls}">
        <div class="sc-score-num c-{risk_cls}">
            {score}<span class="sc-score-denom"> / 100</span>
        </div>
        <div class="sc-score-label c-{risk_cls}">{result.risk_level} RISK</div>
        <div class="sc-score-verdict c-{risk_cls}">{_verdict_label(result.verdict)}</div>
        <div class="sc-risk-bar-wrap" style="margin-top:1rem;">
            <div class="sc-risk-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
        </div>
        <div style="color:#64748b; font-size:0.78rem; margin-top:0.35rem;">
            Confidence: {result.confidence:.0%} &nbsp;·&nbsp; Analysis: {_consistency_label(result.analysis_consistency)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Meta grid (opportunity type, payment context, etc.) ──
    opp_type = _opp_type_label(result.opportunity_type)
    pay_ctx = _payment_ctx_label(result.payment_context)
    consistency = _consistency_label(result.analysis_consistency)

    st.markdown(f"""
    <div class="sc-meta-grid">
        <div class="sc-meta-item">
            <div class="sc-meta-label">Opportunity Type</div>
            <div class="sc-meta-value">{opp_type}</div>
        </div>
        <div class="sc-meta-item">
            <div class="sc-meta-label">Payment Context</div>
            <div class="sc-meta-value">{pay_ctx}</div>
        </div>
        <div class="sc-meta-item">
            <div class="sc-meta-label">Model Confidence</div>
            <div class="sc-meta-value">{result.confidence:.0%}</div>
        </div>
        <div class="sc-meta-item">
            <div class="sc-meta-label">Signal Consistency</div>
            <div class="sc-meta-value">{consistency}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Analysis note ──
    if result.analysis_error:
        st.info(f"ℹ️ {result.analysis_error}")
    if "gemini" not in result.provider_used.lower():
        st.warning("⚠️ Gemini AI unavailable — showing rule-based analysis only. Results may be less accurate.")
    if result.analysis_consistency == "LOW":
        st.warning("⚠️ Analysis signals are inconsistent. Verify this opportunity independently.")

    st.markdown("---")

    # ── Warning indicators ──
    st.markdown('<div class="sc-section-title">Why this was flagged</div>', unsafe_allow_html=True)

    if result.warning_indicators:
        for indicator in result.warning_indicators:
            sev_cls = _severity_css(indicator.severity)
            sev_icon = _severity_icon(indicator.severity)
            title = indicator.title or indicator.type.replace("_", " ").title()
            desc = indicator.description or ""
            source = indicator.source or "Rule Engine"

            st.markdown(f"""
            <div class="sc-indicator sc-ind-{sev_cls}">
                <div class="sc-ind-header">
                    <span>{sev_icon}</span>
                    <span class="sc-ind-title">{title}</span>
                    <span class="sc-badge sc-badge-{sev_cls}">{indicator.severity}</span>
                    <span class="sc-source-tag">{source}</span>
                </div>
                {f'<div class="sc-ind-desc">{desc}</div>' if desc else ''}
                <div class="sc-evidence-box">"{indicator.evidence}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sc-card-tight" style="border-left: 4px solid #16a34a; background:#f0fdf4;">
            <span style="color:#16a34a; font-weight:600;">✅ No significant warning indicators detected.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── AI Assessment + Recommendation (two columns) ──
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="sc-section-title">AI Assessment</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc-card" style="min-height:120px;">
            <p style="color:#334155; line-height:1.75; font-size:0.9rem; margin:0;">{result.explanation}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sc-section-title">What You Should Do</div>', unsafe_allow_html=True)
        rec_text = result.recommendation or "Always verify the opportunity independently through official channels."
        st.markdown(f"""
        <div class="sc-card" style="min-height:120px;">
            <p style="color:#334155; line-height:1.75; font-size:0.9rem; margin:0;">{rec_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Analysis Evidence (expandable) ──
    if result.evidence:
        with st.expander("View Analysis Evidence", expanded=False):
            st.markdown('<div class="sc-section-title">Detected Signals</div>', unsafe_allow_html=True)
            for ev in result.evidence:
                st.markdown(f"""
                <div class="sc-evidence-box" style="margin-bottom:0.5rem;">"{ev}"</div>
                """, unsafe_allow_html=True)

    # ── Technical Analysis (expandable) ──
    with st.expander("Technical Analysis", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Score Breakdown**")
            if result.gemini_score is not None:
                st.markdown(f"🤖 **Gemini AI:** {result.gemini_score:.1f} / 100")
            else:
                st.markdown("🤖 **Gemini AI:** *unavailable*")
            if result.ml_score is not None:
                st.markdown(f"🧪 **Local ML:** {result.ml_score:.1f} / 100")
            else:
                st.markdown("🧪 **Local ML:** *unavailable*")
            if result.rule_score is not None:
                st.markdown(f"📏 **Rule Engine:** {result.rule_score:.1f} / 100")
            st.markdown(f"🎯 **Final Score:** **{result.risk_score:.1f} / 100**")

        with col_b:
            st.markdown("**Analysis Metadata**")
            provider_text = result.provider_used.replace("+", " + ").replace("_", " ").title()
            st.markdown(f"**Providers:** `{provider_text}`")
            st.markdown(f"**Risk Level:** `{result.risk_level}`")
            st.markdown(f"**Verdict:** `{result.verdict}`")
            st.markdown(f"**Opportunity Type:** `{result.opportunity_type}`")
            st.markdown(f"**Payment Context:** `{result.payment_context}`")
            st.markdown(f"**Signal Consistency:** `{result.analysis_consistency}`")
            st.markdown(f"**Confidence:** `{result.confidence:.1%}`")

    # ── Bias disclaimer ──
    st.markdown("""
    <div class="sc-disclaimer">
        AI assessment can produce false positives or false negatives.
        A payment request does not automatically indicate fraud — context matters.
        Always verify the opportunity independently through official channels.
        <br><br>
        <em>ScamCheck is an AI-assisted decision-support tool. It identifies suspicious recruitment indicators but cannot guarantee fraud detection.
        Independently verify employers, organizations and payment requests through official channels.</em>
    </div>
    """, unsafe_allow_html=True)

    # ── Feedback ──
    render_feedback_section(result)


def render_feedback_section(result: AnalysisResult):
    """Render the feedback section at the bottom of results."""
    st.markdown("---")
    st.markdown('<div class="sc-section-title">Was this analysis useful?</div>', unsafe_allow_html=True)

    fb_key = f"fb_state_{id(result)}"
    if fb_key not in st.session_state:
        st.session_state[fb_key] = "pending"

    if st.session_state[fb_key] == "pending":
        col_y, col_n, _ = st.columns([1, 1, 5])
        with col_y:
            if st.button("👍 Helpful", key=f"fb_yes_{fb_key}"):
                saved = save_feedback(
                    feedback_type="HELPFUL",
                    opportunity_type=result.opportunity_type,
                    risk_score=result.risk_score,
                    risk_level=result.risk_level,
                    provider_used=result.provider_used,
                )
                st.session_state[fb_key] = "helpful_done"
                st.rerun()
        with col_n:
            if st.button("👎 Not helpful", key=f"fb_no_{fb_key}"):
                st.session_state[fb_key] = "not_helpful_form"
                st.rerun()

    elif st.session_state[fb_key] == "helpful_done":
        st.success("Thank you for your feedback.")

    elif st.session_state[fb_key] == "not_helpful_form":
        st.markdown("**What could be improved?**")
        reason = st.radio(
            "Reason",
            ["Incorrect risk level", "Incorrect warning", "Missed a warning", "Explanation was unclear", "Other"],
            key=f"fb_reason_{fb_key}",
            label_visibility="collapsed",
        )
        comment = st.text_input(
            "Additional feedback (optional)",
            placeholder="Tell us more...",
            key=f"fb_comment_{fb_key}",
        )
        reason_codes = {
            "Incorrect risk level": "INCORRECT_RISK_LEVEL",
            "Incorrect warning": "INCORRECT_WARNING",
            "Missed a warning": "MISSED_WARNING",
            "Explanation was unclear": "UNCLEAR_EXPLANATION",
            "Other": "OTHER",
        }
        if st.button("Submit Feedback", key=f"fb_submit_{fb_key}"):
            save_feedback(
                feedback_type="NOT_HELPFUL",
                reason=reason_codes.get(reason, "OTHER"),
                comment=comment or None,
                opportunity_type=result.opportunity_type,
                risk_score=result.risk_score,
                risk_level=result.risk_level,
                provider_used=result.provider_used,
            )
            st.session_state[fb_key] = "not_helpful_done"
            st.rerun()

    elif st.session_state[fb_key] == "not_helpful_done":
        st.info("Thank you — your feedback has been recorded.")


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_analyze():
    # ── Header ──
    service = get_analysis_service()
    try:
        status = service.get_provider_status()
        gemini_ok = status["gemini"]["available"]
        ml_ok = status["local_ml"]["available"]
        ai_online = gemini_ok or ml_ok
    except Exception:
        gemini_ok = ml_ok = ai_online = False

    if ai_online:
        status_html = '<span class="sc-status-online"><span class="sc-status-dot-green"></span> AI Engine Online</span>'
    else:
        status_html = '<span class="sc-status-offline"><span class="sc-status-dot-red"></span> AI Engine Offline</span>'

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">
        <div>
            <div class="sc-brand">🛡️ ScamCheck</div>
            <div class="sc-subtitle">AI-Powered Opportunity Verification</div>
        </div>
        <div>{status_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Hero ──
    st.markdown("""
    <div class="sc-hero">
        <div class="sc-hero-title">Verify before you commit.</div>
        <div class="sc-hero-sub">
            Check internships, jobs and recruitment messages<br>
            before you pay, click, or share sensitive information.
        </div>
        <div class="sc-feature-pills">
            <span class="sc-pill">🤖 AI-assisted analysis</span>
            <span class="sc-pill">🔍 Explainable risk indicators</span>
            <span class="sc-pill">🔒 Privacy-conscious design</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample inputs ──
    scam_examples, legit_examples = load_demo_examples()
    all_examples = (
        [(f"🔴 {e['title']}", e['text']) for e in scam_examples] +
        [(f"✅ {e['title']}", e['text']) for e in legit_examples]
    )

    if all_examples:
        st.markdown('<div style="font-size:0.8rem; color:#64748b; font-weight:600; margin-bottom:0.4rem;">Try a sample:</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(all_examples), 5))
        for i, (label, text) in enumerate(all_examples[:5]):
            with cols[i % 5]:
                if st.button(label, key=f"sample_{i}"):
                    st.session_state["demo_text"] = text
                    st.rerun()

    st.markdown("---")

    # ── Input Tabs ──
    tab_paste, tab_structured, tab_screenshot = st.tabs([
        "Message",
        "Job Details",
        "Screenshot",
    ])

    # ─── Tab 1: Paste Message ───
    with tab_paste:
        st.markdown('<p style="color:#64748b; font-size:0.875rem; margin-bottom:0.5rem;">Paste a WhatsApp message, email, or social media post about a job or internship opportunity.</p>', unsafe_allow_html=True)

        default_text = st.session_state.pop("demo_text", "")

        text_input = st.text_area(
            "Opportunity Message",
            value=default_text,
            placeholder="Paste the opportunity message you received from WhatsApp, email or social media...",
            height=200,
            key="paste_text",
            label_visibility="collapsed",
        )

        char_count = len(text_input)
        char_cls = "sc-char-warn" if char_count > MAX_INPUT_CHARS * 0.9 else ""
        st.markdown(f'<div class="sc-char-count {char_cls}">{char_count} / {MAX_INPUT_CHARS} characters</div>', unsafe_allow_html=True)

        if st.button("Analyze Opportunity", key="analyze_paste", use_container_width=True):
            if not text_input.strip():
                st.error("Please provide an opportunity message to analyze.")
            elif char_count > MAX_INPUT_CHARS:
                st.warning(f"Message truncated to {MAX_INPUT_CHARS} characters for analysis.")
                text_to_analyze = text_input[:MAX_INPUT_CHARS]
                _run_text_analysis(service, text_to_analyze)
            else:
                is_valid, error_msg = validate_text_input(text_input)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    _run_text_analysis(service, text_input)

    # ─── Tab 2: Job Details ───
    with tab_structured:
        st.markdown('<p style="color:#64748b; font-size:0.875rem; margin-bottom:1rem;">Fill in the details of the opportunity you received. Leave fields blank if unknown.</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company Name", placeholder="e.g., Google, TCS, Startup XYZ", key="s_company")
            role = st.text_input("Job / Internship Role", placeholder="e.g., Software Engineer Intern", key="s_role")
            salary = st.text_input("Salary / Stipend", placeholder="e.g., ₹15,000/month or ₹5 LPA", key="s_salary")
            emp_type = st.selectbox("Employment Type", ["Unknown", "Full-time", "Part-time", "Internship", "Freelance", "Remote"], key="s_emptype")

        with col2:
            reg_fee = st.text_input("Registration Fee", placeholder="e.g., ₹2,999 or None", key="s_fee")
            app_fee = st.text_input("Application Fee", placeholder="e.g., ₹500 or None", key="s_appfee")
            contact = st.text_input("Contact Method", placeholder="e.g., WhatsApp, official email, LinkedIn", key="s_contact")
            website = st.text_input("Website / Link", placeholder="e.g., https://company.com/jobs", key="s_website")

        description = st.text_area(
            "Message / Opportunity Description",
            placeholder="Paste the full offer message or describe the opportunity...",
            height=120,
            key="s_desc",
        )

        if st.button("Analyze Opportunity", key="analyze_structured", use_container_width=True):
            has_content = any([company, role, salary, reg_fee, app_fee, contact, website, description])
            if not has_content:
                st.error("Please fill in at least some details about the opportunity.")
            else:
                with st.spinner("Analyzing opportunity details..."):
                    try:
                        result = service.analyze_structured(
                            company=company,
                            role=role,
                            salary=salary,
                            registration_fee=reg_fee,
                            application_fee=app_fee,
                            contact_method=contact,
                            website=website,
                            description=description,
                        )
                        st.markdown("---")
                        render_result(result)
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    # ─── Tab 3: Screenshot ───
    with tab_screenshot:
        st.markdown('<p style="color:#64748b; font-size:0.875rem; margin-bottom:1rem;">Upload a screenshot of the job offer, WhatsApp message, or email.</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key="screenshot_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Screenshot", use_column_width=True)

            extra_context = st.text_input(
                "Additional Context (optional)",
                placeholder="e.g., Received this on WhatsApp from an unknown number",
                key="screenshot_context",
            )

            if st.button("Analyze Screenshot", key="analyze_screenshot", use_container_width=True):
                with st.spinner("Analyzing screenshot with AI vision..."):
                    try:
                        result = service.analyze_image(image, context=extra_context or None)
                        st.markdown("---")
                        render_result(result)
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        else:
            st.markdown("""
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2.5rem 2rem; text-align: center; color: #94a3b8; background:#f8fafc;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📸</div>
                <div style="font-weight: 500; color:#475569;">Upload a screenshot of the opportunity</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #94a3b8;">Supported: PNG · JPG · JPEG · WebP</div>
            </div>
            """, unsafe_allow_html=True)


def _run_text_analysis(service, text: str):
    with st.spinner("Analyzing opportunity..."):
        try:
            result = service.analyze_text(text)
            st.markdown("---")
            render_result(result)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}\n\nPlease check your API configuration.")


def page_how_it_works():
    st.markdown("""
    <div class="sc-brand" style="margin-bottom:0.25rem;">⚙️ How It Works</div>
    <div class="sc-subtitle" style="margin-bottom:1.5rem;">Understanding ScamCheck's analysis pipeline</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    steps = [
        ("Submit", "Paste a recruitment message, fill in job details, or upload a screenshot of the opportunity."),
        ("Classify", "AI identifies the opportunity type (job, internship, paid training, certification) and whether any payment is for employment or a legitimate service."),
        ("Analyze", "Three engines run in parallel: Gemini AI (contextual reasoning), Local ML classifier (dataset-based), and the Rule Engine (deterministic pattern matching)."),
        ("Score", "A weighted risk score (0–100) is calculated. Payment context determines whether financial signals contribute to the risk or not."),
        ("Explain", "Warning indicators, evidence, and a human-readable explanation are generated — with source attribution for each signal."),
    ]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="sc-step">
            <div class="sc-step-num">{i}</div>
            <div class="sc-step-content">
                <div class="sc-step-title">{title}</div>
                <div class="sc-step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk scale
    st.markdown('<div class="sc-section-title">Risk Scale</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    risk_levels = [
        ("0–24", "LOW", "#16a34a", "#f0fdf4", "#86efac", "Appears legitimate. Standard caution advised."),
        ("25–49", "MEDIUM", "#d97706", "#fffbeb", "#fcd34d", "Some concerns. Verify independently."),
        ("50–74", "HIGH", "#ea580c", "#fff7ed", "#fdba74", "Multiple red flags. Likely problematic."),
        ("75–100", "CRITICAL", "#dc2626", "#fef2f2", "#fca5a5", "Strong scam indicators. Do not engage."),
    ]
    for col, (rng, label, color, bg, border, desc) in zip(cols, risk_levels):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; background:{bg}; border-radius:10px; border:1px solid {border};">
                <div style="color:{color}; font-weight:700; font-size:1.1rem;">{rng}</div>
                <div style="color:{color}; font-weight:700; font-size:0.9rem; margin:0.25rem 0;">{label}</div>
                <div style="color:#64748b; font-size:0.78rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sc-disclaimer">
        ScamCheck provides decision support — not guaranteed fraud detection.
        Always verify opportunities through official company channels.
    </div>
    """, unsafe_allow_html=True)


def page_model_info():
    st.markdown("""
    <div class="sc-brand" style="margin-bottom:0.25rem;">🔬 Model Information</div>
    <div class="sc-subtitle" style="margin-bottom:1.5rem;">Hybrid AI analysis — provider status and system configuration</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    service = get_analysis_service()
    status = service.get_provider_status()

    st.markdown('<div class="sc-section-title">Hybrid Analysis Architecture</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gemini_ok = status["gemini"]["available"]
        badge = "✅ Active" if gemini_ok else "❌ Unavailable"
        badge_color = "#16a34a" if gemini_ok else "#dc2626"
        err = status["gemini"].get("error") or ""
        st.markdown(f"""
        <div class="sc-card">
            <div style="font-weight:700; color:#0f172a; font-size:1rem;">🤖 Gemini AI</div>
            <div style="color:{badge_color}; font-weight:600; font-size:0.85rem; margin:0.4rem 0;">{badge}</div>
            <p style="color:#64748b; font-size:0.82rem; line-height:1.6;">
                Contextual reasoning using Google's Gemini model.
                Classifies opportunity type, payment context, and generates
                structured risk indicators and explanations.
                Also supports screenshot analysis via multimodal vision.
            </p>
            {f'<p style="color:#dc2626; font-size:0.78rem;">{err}</p>' if not gemini_ok and err else ''}
            <p style="color:#94a3b8; font-size:0.75rem;">Model: {status["gemini"]["name"]}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ml_ok = status["local_ml"]["available"]
        badge2 = "✅ Trained & Active" if ml_ok else "⚠️ Not Trained"
        badge_color2 = "#16a34a" if ml_ok else "#d97706"
        ml_err = status["local_ml"].get("error") or ""
        st.markdown(f"""
        <div class="sc-card">
            <div style="font-weight:700; color:#0f172a; font-size:1rem;">🧪 Local ML Classifier</div>
            <div style="color:{badge_color2}; font-weight:600; font-size:0.85rem; margin:0.4rem 0;">{badge2}</div>
            <p style="color:#64748b; font-size:0.82rem; line-height:1.6;">
                A locally trained supervised classifier (TF-IDF + Logistic Regression)
                provides a dataset-based scam probability signal.
                We use a locally trained classifier alongside Gemini's contextual
                reasoning — they are not the same model.
            </p>
            {f'<p style="color:#d97706; font-size:0.78rem;">{ml_err}</p>' if not ml_ok and ml_err else ''}
            <p style="color:#94a3b8; font-size:0.75rem;">Train: <code>python scripts/train_model.py</code></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="sc-card">
            <div style="font-weight:700; color:#0f172a; font-size:1rem;">📏 Rule Engine</div>
            <div style="color:#16a34a; font-weight:600; font-size:0.85rem; margin:0.4rem 0;">✅ Always Active</div>
            <p style="color:#64748b; font-size:0.82rem; line-height:1.6;">
                Deterministic pattern matching for high-confidence scam signals:
                employment payment requests, OTP/credential demands,
                suspicious URLs, guaranteed-job claims, and pressure tactics.
                Runs even when AI providers are offline.
            </p>
            <p style="color:#94a3b8; font-size:0.75rem;">Context-aware: payment type modulates score contribution.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sc-section-title">Quick Setup</div>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Gemini API key
cp .env.example .env
# Edit .env: add GEMINI_API_KEY=your_key_here

# 3. Train local ML model (optional)
python scripts/train_model.py

# 4. Run
streamlit run app.py
""", language="bash")


def page_privacy():
    st.markdown("""
    <div class="sc-brand" style="margin-bottom:0.25rem;">🔒 Privacy</div>
    <div class="sc-subtitle" style="margin-bottom:1.5rem;">How ScamCheck handles your data</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="sc-card">
        <div class="sc-section-title">What is processed</div>
        <p style="color:#334155; font-size:0.9rem; line-height:1.75;">
            When you submit a message or screenshot for analysis, its content is sent to the configured AI provider
            (Google Gemini) for contextual analysis. This is necessary to generate an accurate risk assessment.
            The local ML classifier and rule engine process your text entirely on-device.
        </p>
        <p style="color:#334155; font-size:0.9rem; line-height:1.75; margin-top:0.75rem;">
            <strong>Avoid submitting passwords, OTPs, bank credentials, Aadhaar numbers, or other highly sensitive
            personal information in this tool.</strong> Although ScamCheck does not store submitted content,
            data sent to external AI APIs is subject to the provider's privacy policy.
        </p>
    </div>

    <div class="sc-card">
        <div class="sc-section-title">Feedback data</div>
        <p style="color:#334155; font-size:0.9rem; line-height:1.75;">
            If you submit feedback, only the following is stored locally:
            feedback type, optional reason, optional comment, opportunity type, risk score, and provider used.
            <strong>Your submitted message is never stored as part of feedback.</strong>
        </p>
    </div>

    <div class="sc-card">
        <div class="sc-section-title">What ScamCheck does NOT do</div>
        <ul style="color:#334155; font-size:0.9rem; line-height:1.9; margin-top:0.5rem;">
            <li>Does not store your submitted messages or screenshots</li>
            <li>Does not collect personal information</li>
            <li>Does not require login or account creation</li>
            <li>Does not share data with third parties beyond the configured AI provider</li>
            <li>Does not automatically open URLs in submitted content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.25rem 0 0.75rem;">
            <div style="font-size:1.3rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em;">🛡️ ScamCheck</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.15rem; font-weight:500;">Opportunity Verification · Hackspora 2.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        nav = st.radio(
            "Navigation",
            ["Analyze", "How It Works", "Model", "Privacy"],
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Provider status
        st.markdown('<div style="color:#94a3b8; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">Provider Status</div>', unsafe_allow_html=True)

        try:
            service = get_analysis_service()
            status = service.get_provider_status()
            gemini_ok = status["gemini"]["available"]
            ml_ok = status["local_ml"]["available"]

            g_color = "#16a34a" if gemini_ok else "#dc2626"
            m_color = "#16a34a" if ml_ok else "#d97706"
            g_icon = "●" if gemini_ok else "○"
            m_icon = "●" if ml_ok else "○"

            st.markdown(f"""
            <div style="font-size:0.82rem; color:{g_color}; margin-bottom:0.3rem; font-weight:500;">{g_icon} Gemini AI</div>
            <div style="font-size:0.82rem; color:{m_color}; margin-bottom:0.3rem; font-weight:500;">{m_icon} Local ML Model</div>
            <div style="font-size:0.82rem; color:#16a34a; font-weight:500;">● Rule Engine</div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown('<div style="color:#dc2626; font-size:0.8rem;">Status unavailable</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#94a3b8; font-size:0.75rem; line-height:1.7;">
            <strong style="color:#64748b;">Stack</strong><br>
            Streamlit · Python<br>
            Google Gemini API<br>
            scikit-learn · TF-IDF
        </div>
        """, unsafe_allow_html=True)

    return nav


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    nav = render_sidebar()

    if nav == "Analyze":
        page_analyze()
    elif nav == "How It Works":
        page_how_it_works()
    elif nav == "Model":
        page_model_info()
    elif nav == "Privacy":
        page_privacy()


if __name__ == "__main__":
    main()
