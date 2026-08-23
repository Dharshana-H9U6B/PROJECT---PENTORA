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
# Add project root to Python path
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
from backend.schemas import AnalysisResult, RiskLevel
from backend.input_processors.text_processor import validate_text_input, build_structured_text


# ── Styling ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #0a1220 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }

    /* Risk Cards */
    .risk-card {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    .risk-score-display {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .score-number {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
    }

    .score-label {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: 0.1em;
    }

    /* Risk level colors */
    .risk-critical { background: linear-gradient(135deg, rgba(220,38,38,0.25), rgba(153,27,27,0.15)); border: 2px solid rgba(239,68,68,0.6); }
    .risk-high { background: linear-gradient(135deg, rgba(217,119,6,0.25), rgba(180,83,9,0.15)); border: 2px solid rgba(245,158,11,0.6); }
    .risk-medium { background: linear-gradient(135deg, rgba(202,138,4,0.25), rgba(161,98,7,0.15)); border: 2px solid rgba(234,179,8,0.6); }
    .risk-low { background: linear-gradient(135deg, rgba(22,163,74,0.25), rgba(21,128,61,0.15)); border: 2px solid rgba(34,197,94,0.6); }

    .color-critical { color: #f87171; }
    .color-high { color: #fbbf24; }
    .color-medium { color: #fde047; }
    .color-low { color: #4ade80; }

    /* Indicator cards */
    .indicator-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
    }
    .ind-critical { border-left-color: #f87171; }
    .ind-high { border-left-color: #fbbf24; }
    .ind-medium { border-left-color: #fde047; }
    .ind-low { border-left-color: #4ade80; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
    }

    /* Demo button */
    .demo-btn > button {
        background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
        box-shadow: 0 4px 15px rgba(139,92,246,0.3) !important;
    }

    /* Text inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.4rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Warning banner */
    .warning-banner {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #fbbf24;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }

    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Evidence text */
    .evidence-text {
        background: rgba(0,0,0,0.3);
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.875rem;
        font-family: 'Inter', monospace;
        color: #cbd5e1;
        margin-top: 0.3rem;
        border-left: 3px solid rgba(99,102,241,0.5);
    }

    /* Provider status */
    .provider-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-active { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .badge-inactive { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

    /* Score breakdown */
    .score-breakdown-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #818cf8;
        font-size: 0.8rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_risk_color_class(risk_level: str) -> str:
    mapping = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
    return mapping.get(risk_level, "low")


def get_risk_emoji(risk_level: str) -> str:
    mapping = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🟡", "LOW": "✅"}
    return mapping.get(risk_level, "✅")


def get_severity_emoji(severity: str) -> str:
    mapping = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    return mapping.get(severity, "⚪")


def get_verdict_label(verdict: str) -> str:
    labels = {
        "LIKELY_LEGIT": "Likely Legitimate",
        "SUSPICIOUS": "Suspicious",
        "POTENTIAL_SCAM": "Potential Scam",
        "HIGH_RISK_SCAM": "High Risk Scam",
    }
    return labels.get(verdict, verdict)


def load_demo_examples() -> tuple[list, list]:
    """Load demo examples from data/demo/."""
    demo_dir = ROOT / "data" / "demo"
    scam_examples = []
    legit_examples = []

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


# ── Result rendering ───────────────────────────────────────────────────────────

def render_result(result: AnalysisResult):
    """Render the complete analysis result UI."""
    risk_class = get_risk_color_class(result.risk_level)
    risk_emoji = get_risk_emoji(result.risk_level)
    score = int(result.risk_score)

    # ── Risk Score Card ──
    st.markdown(f"""
    <div class="risk-score-display risk-{risk_class}">
        <div class="score-number color-{risk_class}">{score}</div>
        <div style="color: #94a3b8; font-size: 1rem; margin: 0.25rem 0;">out of 100</div>
        <div class="score-label color-{risk_class}">{risk_emoji} {result.risk_level} RISK</div>
        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
            {get_verdict_label(result.verdict)} &nbsp;·&nbsp;
            Confidence: {result.confidence:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress bar ──
    st.progress(score / 100)

    # ── Warning / Error notices ──
    if result.analysis_error:
        st.warning(f"⚠️ Analysis note: {result.analysis_error}")

    provider_text = result.provider_used.replace("+", " + ").replace("_", " ").title()
    if "gemini" not in result.provider_used.lower():
        st.warning("⚠️ Gemini analysis unavailable. Showing local model + rule-based analysis only.")

    # ── Warning Indicators ──
    if result.warning_indicators:
        st.markdown('<div class="section-header">🚩 Warning Indicators</div>', unsafe_allow_html=True)

        for indicator in result.warning_indicators:
            sev_class = indicator.severity.lower()
            sev_emoji = get_severity_emoji(indicator.severity)
            ind_title = indicator.type.replace("_", " ").title()
            desc = indicator.description or ""

            st.markdown(f"""
            <div class="indicator-card ind-{sev_class}">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                    <span>{sev_emoji}</span>
                    <span style="color:#e2e8f0; font-weight:600;">{ind_title}</span>
                    <span style="color:#64748b; font-size:0.8rem; margin-left:auto;">{indicator.severity}</span>
                </div>
                {f'<div style="color:#94a3b8; font-size:0.875rem; margin-bottom:0.3rem;">{desc}</div>' if desc else ''}
                <div class="evidence-text">"{indicator.evidence}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="indicator-card ind-low">
            <div style="color:#4ade80; font-weight:600;">✅ No significant warning indicators detected.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Two columns: Explanation + Recommendation ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🧠 AI Explanation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card" style="padding: 1.2rem;">
            <p style="color:#cbd5e1; line-height:1.7; font-size:0.95rem;">{result.explanation}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">💡 Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card" style="padding: 1.2rem;">
            <p style="color:#cbd5e1; line-height:1.7; font-size:0.95rem;">{result.recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Score Breakdown (expandable) ──
    with st.expander("📊 Analysis Details", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Score Breakdown**")
            if result.gemini_score is not None:
                st.markdown(f"🤖 Gemini Score: **{result.gemini_score:.1f} / 100**")
            else:
                st.markdown("🤖 Gemini Score: *unavailable*")

            if result.ml_score is not None:
                st.markdown(f"🧪 ML Score: **{result.ml_score:.1f} / 100**")
            else:
                st.markdown("🧪 ML Score: *unavailable*")

            if result.rule_score is not None:
                st.markdown(f"📏 Rule Score: **{result.rule_score:.1f} / 100**")

            st.markdown(f"🎯 **Final Score: {result.risk_score:.1f} / 100**")

        with col_b:
            st.markdown("**Providers Used**")
            st.markdown(f"`{provider_text}`")
            st.markdown(f"**Risk Level:** `{result.risk_level}`")
            st.markdown(f"**Verdict:** `{result.verdict}`")
            st.markdown(f"**Confidence:** `{result.confidence:.1%}`")

        if result.evidence:
            st.markdown("**Key Evidence**")
            for ev in result.evidence:
                st.markdown(f"• {ev}")

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> ScamCheck is an AI-powered decision support tool.
        It identifies recruitment scam indicators and produces an interpretable risk score.
        It does not guarantee fraud detection. Always verify opportunities independently
        through official company channels before taking any action.
    </div>
    """, unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_analyze():
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🛡️ ScamCheck</div>
        <div class="app-subtitle">AI-Powered Internship & Job Opportunity Verification</div>
    </div>
    """, unsafe_allow_html=True)

    service = get_analysis_service()

    # ── Demo Examples Selector ──
    scam_examples, legit_examples = load_demo_examples()
    all_examples = (
        [(f"🔴 {e['title']}", e['text']) for e in scam_examples] +
        [(f"✅ {e['title']}", e['text']) for e in legit_examples]
    )

    with st.expander("🎯 Load Demo Example", expanded=False):
        st.markdown('<span style="color:#94a3b8; font-size:0.875rem;">Select a built-in example to try the system without needing a real message.</span>', unsafe_allow_html=True)
        demo_options = ["— Select a demo example —"] + [label for label, _ in all_examples]
        selected_demo = st.selectbox("Choose an example:", demo_options, key="demo_selector", label_visibility="collapsed")

        if selected_demo != "— Select a demo example —":
            demo_text = next(text for label, text in all_examples if label == selected_demo)
            if st.button("📋 Load This Example", key="load_demo"):
                st.session_state["demo_text"] = demo_text
                st.rerun()

    st.markdown("---")

    # ── Input Tabs ──
    tab_paste, tab_structured, tab_screenshot = st.tabs([
        "📝 Paste Message",
        "📋 Job Details",
        "📸 Screenshot",
    ])

    # ─── Tab 1: Paste Message ───
    with tab_paste:
        st.markdown('<p style="color:#94a3b8; margin-bottom:0.5rem;">Paste a WhatsApp message, email, or social media post about a job/internship opportunity.</p>', unsafe_allow_html=True)

        # Populate from demo if selected
        default_text = st.session_state.pop("demo_text", "")

        text_input = st.text_area(
            "Opportunity Message",
            value=default_text,
            placeholder="Paste WhatsApp, email or social-media opportunity here...\n\nExample: Congratulations! You have been selected for a Google internship. Pay ₹2,999 registration fee today to confirm your position...",
            height=200,
            key="paste_text",
            label_visibility="collapsed",
        )

        if st.button("🔍 Analyze Opportunity", key="analyze_paste", use_container_width=True):
            is_valid, error_msg = validate_text_input(text_input)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                with st.spinner("🔍 Analyzing opportunity..."):
                    try:
                        result = service.analyze_text(text_input)
                        st.markdown("---")
                        render_result(result)
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}\n\nPlease check your API key configuration.")

    # ─── Tab 2: Structured Form ───
    with tab_structured:
        st.markdown('<p style="color:#94a3b8; margin-bottom:1rem;">Fill in the details of the job or internship opportunity you received.</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("🏢 Company Name", placeholder="e.g., Google, TCS, Startup XYZ", key="s_company")
            role = st.text_input("💼 Job / Internship Role", placeholder="e.g., Software Engineer Intern", key="s_role")
            salary = st.text_input("💰 Salary / Stipend", placeholder="e.g., ₹15,000/month or ₹5 LPA", key="s_salary")
            reg_fee = st.text_input("🔴 Registration / Joining Fee", placeholder="e.g., ₹2,999 or None", key="s_fee")

        with col2:
            contact = st.text_input("📱 Contact Method", placeholder="e.g., WhatsApp, email, LinkedIn", key="s_contact")
            website = st.text_input("🌐 Website / Link", placeholder="e.g., https://company.com/jobs", key="s_website")

        description = st.text_area(
            "📄 Message / Description",
            placeholder="Paste the full offer message or describe the opportunity...",
            height=120,
            key="s_desc",
        )

        if st.button("🔍 Analyze Opportunity", key="analyze_structured", use_container_width=True):
            has_content = any([company, role, salary, reg_fee, contact, website, description])
            if not has_content:
                st.error("❌ Please fill in at least some details about the opportunity.")
            else:
                with st.spinner("🔍 Analyzing opportunity details..."):
                    try:
                        result = service.analyze_structured(
                            company=company,
                            role=role,
                            salary=salary,
                            registration_fee=reg_fee,
                            contact_method=contact,
                            website=website,
                            description=description,
                        )
                        st.markdown("---")
                        render_result(result)
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

    # ─── Tab 3: Screenshot ───
    with tab_screenshot:
        st.markdown('<p style="color:#94a3b8; margin-bottom:1rem;">Upload a screenshot of the job offer, WhatsApp message, or email.</p>', unsafe_allow_html=True)

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

            if st.button("🔍 Analyze Screenshot", key="analyze_screenshot", use_container_width=True):
                with st.spinner("🔍 Analyzing screenshot with AI vision..."):
                    try:
                        result = service.analyze_image(image, context=extra_context or None)
                        st.markdown("---")
                        render_result(result)
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(99,102,241,0.3); border-radius: 12px; padding: 3rem 2rem; text-align: center; color: #64748b;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
                <div>Drag & drop a screenshot, or click <strong>Browse files</strong> above.</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #475569;">Supported: PNG, JPG, JPEG, WebP</div>
            </div>
            """, unsafe_allow_html=True)


def page_how_it_works():
    st.markdown("""
    <div class="app-header">
        <div class="app-title" style="font-size:2.5rem;">⚙️ How It Works</div>
        <div class="app-subtitle">Understanding ScamCheck's analysis pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="risk-card">
            <h3 style="color:#60a5fa; margin-bottom:1rem;">🤖 Gemini AI Analysis</h3>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem;">
                Google's Gemini model reads the full opportunity message and uses contextual
                reasoning to identify fraud patterns. It generates structured JSON with a
                risk score, warning indicators, and human-readable explanations.
            </p>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem; margin-top:0.5rem;">
                For screenshots, Gemini uses its multimodal vision capability to read
                and analyze image content directly.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="risk-card">
            <h3 style="color:#818cf8; margin-bottom:1rem;">🧪 Local ML Classifier</h3>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem;">
                A TF-IDF + Logistic Regression model trained on a scam/legitimate dataset
                provides a second signal. It transforms text into numerical features and
                predicts a scam probability.
            </p>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem; margin-top:0.5rem;">
                Train your own model with any CSV dataset using:
                <br><code style="color:#818cf8;">python scripts/train_model.py</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="risk-card">
            <h3 style="color:#a78bfa; margin-bottom:1rem;">📏 Rule-Based Detection</h3>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem;">
                Deterministic pattern matching catches known scam signals:
                upfront payment requests, urgency tactics, suspicious URLs,
                OTP/banking credential requests, and fake company claims.
            </p>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.9rem; margin-top:0.5rem;">
                Rules always run — they work even when AI providers are offline.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="risk-card">
        <h3 style="color:#e2e8f0; margin-bottom:1rem;">🎯 Risk Aggregation</h3>
        <p style="color:#94a3b8; line-height:1.7;">
            The three signals are combined using configurable weights (default: Gemini 50%, ML 30%, Rules 20%).
            If a provider is unavailable, its weight is redistributed to the remaining ones.
            The final score (0–100) determines the risk level:
        </p>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-top:1rem;">
            <div style="text-align:center; padding:1rem; background:rgba(34,197,94,0.1); border-radius:10px; border:1px solid rgba(34,197,94,0.3);">
                <div style="color:#4ade80; font-weight:700; font-size:1.2rem;">0–24</div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.3rem;">LOW<br>Likely Legitimate</div>
            </div>
            <div style="text-align:center; padding:1rem; background:rgba(234,179,8,0.1); border-radius:10px; border:1px solid rgba(234,179,8,0.3);">
                <div style="color:#fde047; font-weight:700; font-size:1.2rem;">25–49</div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.3rem;">MEDIUM<br>Suspicious</div>
            </div>
            <div style="text-align:center; padding:1rem; background:rgba(245,158,11,0.1); border-radius:10px; border:1px solid rgba(245,158,11,0.3);">
                <div style="color:#fbbf24; font-weight:700; font-size:1.2rem;">50–74</div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.3rem;">HIGH<br>Potential Scam</div>
            </div>
            <div style="text-align:center; padding:1rem; background:rgba(239,68,68,0.1); border-radius:10px; border:1px solid rgba(239,68,68,0.3);">
                <div style="color:#f87171; font-weight:700; font-size:1.2rem;">75–100</div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.3rem;">CRITICAL<br>High Risk Scam</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ScamCheck provides decision support — not guaranteed fraud detection.
        Always verify opportunities through official company channels.
    </div>
    """, unsafe_allow_html=True)


def page_model_info():
    st.markdown("""
    <div class="app-header">
        <div class="app-title" style="font-size:2.5rem;">🔬 Model Information</div>
        <div class="app-subtitle">Provider status and system configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    service = get_analysis_service()
    status = service.get_provider_status()

    col1, col2 = st.columns(2)

    with col1:
        gemini_ok = status["gemini"]["available"]
        gemini_icon = "✅" if gemini_ok else "❌"
        gemini_badge = "badge-active" if gemini_ok else "badge-inactive"
        gemini_status = "Active" if gemini_ok else "Unavailable"
        gemini_error = status["gemini"]["error"] or ""

        st.markdown(f"""
        <div class="risk-card">
            <h3 style="color:#60a5fa;">{gemini_icon} Gemini AI Provider</h3>
            <div style="margin: 0.75rem 0;">
                <span class="provider-badge {gemini_badge}">{gemini_status}</span>
            </div>
            <p style="color:#94a3b8; font-size:0.875rem;">
                <strong style="color:#cbd5e1;">Model:</strong> {status["gemini"]["name"]}
            </p>
            {"<p style='color:#f87171; font-size:0.8rem; margin-top:0.5rem;'>⚠️ " + gemini_error + "</p>" if not gemini_ok and gemini_error else ""}
            <p style="color:#64748b; font-size:0.8rem; margin-top:0.75rem;">
                Capabilities: Text analysis, Screenshot analysis (multimodal),
                Structured form analysis, Contextual scam indicator detection.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ml_ok = status["local_ml"]["available"]
        ml_icon = "✅" if ml_ok else "⚠️"
        ml_badge = "badge-active" if ml_ok else "badge-inactive"
        ml_status = "Trained & Active" if ml_ok else "Not Trained"
        ml_error = status["local_ml"]["error"] or ""

        st.markdown(f"""
        <div class="risk-card">
            <h3 style="color:#818cf8;">{ml_icon} Local ML Provider</h3>
            <div style="margin: 0.75rem 0;">
                <span class="provider-badge {ml_badge}">{ml_status}</span>
            </div>
            <p style="color:#94a3b8; font-size:0.875rem;">
                <strong style="color:#cbd5e1;">Type:</strong> TF-IDF + Logistic Regression
            </p>
            {"<p style='color:#fbbf24; font-size:0.8rem; margin-top:0.5rem;'>ℹ️ " + ml_error + "</p>" if not ml_ok and ml_error else ""}
            <p style="color:#64748b; font-size:0.8rem; margin-top:0.75rem;">
                Train with: <code>python scripts/train_model.py</code><br>
                Supports any CSV dataset via config.yaml configuration.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="risk-card">
        <h3 style="color:#e2e8f0; margin-bottom:1rem;">⚙️ Quick Setup Guide</h3>
    </div>
    """, unsafe_allow_html=True)

    st.code("""# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Train the local ML model (optional but recommended)
python scripts/train_model.py

# 4. Run the application
streamlit run app.py
""", language="bash")

    st.markdown("""
    <div class="risk-card" style="margin-top:1rem;">
        <h3 style="color:#e2e8f0; margin-bottom:1rem;">🔄 Using a Custom Dataset</h3>
        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
            Place any CSV file in <code>data/raw/</code> and update <code>config.yaml</code>:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""# config.yaml
dataset:
  path: data/raw/your_dataset.csv
  text_column: message      # column containing the text
  label_column: label       # column containing the label
  labels:
    scam: [scam, spam, fraud, "1"]
    legitimate: [legitimate, ham, safe, "0"]

# Then retrain:
# python scripts/train_model.py
""", language="yaml")

    st.markdown("""
    <div class="risk-card" style="margin-top:1rem;">
        <h3 style="color:#e2e8f0; margin-bottom:1rem;">🔌 Switching AI Providers</h3>
        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
            To add a new provider (e.g., HuggingFace), create a new class in
            <code>backend/models/</code> that extends <code>ScamAnalysisProvider</code>:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""# backend/models/huggingface_provider.py
from backend.models.base import ScamAnalysisProvider
from backend.schemas import AnalysisResult

class HuggingFaceProvider(ScamAnalysisProvider):
    def analyze_text(self, text: str) -> AnalysisResult:
        # Your implementation here
        ...
""", language="python")


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">ScamCheck</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:0.25rem;">v1.0 · Hackspora 2.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        nav = st.radio(
            "Navigation",
            ["🔍 Analyze Opportunity", "⚙️ How It Works", "🔬 Model Information"],
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Provider status mini-display
        st.markdown('<div style="color:#64748b; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Provider Status</div>', unsafe_allow_html=True)

        try:
            service = get_analysis_service()
            status = service.get_provider_status()
            gemini_ok = status["gemini"]["available"]
            ml_ok = status["local_ml"]["available"]

            st.markdown(f"""
            <div style="font-size:0.8rem; color:{'#4ade80' if gemini_ok else '#f87171'}; margin-bottom:0.3rem;">
                {'✅' if gemini_ok else '❌'} Gemini AI
            </div>
            <div style="font-size:0.8rem; color:{'#4ade80' if ml_ok else '#fbbf24'};">
                {'✅' if ml_ok else '⚠️'} Local ML Model
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown('<div style="color:#f87171; font-size:0.8rem;">Status unavailable</div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="color:#475569; font-size:0.75rem; line-height:1.6;">
            <strong style="color:#64748b;">Built for</strong><br>
            Hackspora 2.0<br>
            Problem Statement 3<br><br>
            <strong style="color:#64748b;">Stack</strong><br>
            Streamlit · Python<br>
            Google Gemini API<br>
            scikit-learn · TF-IDF
        </div>
        """, unsafe_allow_html=True)

    return nav


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    nav = render_sidebar()

    if "🔍 Analyze Opportunity" in nav:
        page_analyze()
    elif "⚙️ How It Works" in nav:
        page_how_it_works()
    elif "🔬 Model Information" in nav:
        page_model_info()


if __name__ == "__main__":
    main()
