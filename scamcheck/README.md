# ScamCheck — AI-Powered Internship & Job Opportunity Verification

> **Hackspora 2.0 · Problem Statement 3**

ScamCheck is an AI-powered decision support tool that helps students identify suspicious internship and job opportunities. It analyzes submitted opportunity details and generates an interpretable risk score with clear warning indicators.

---

## Features

- **Multi-input analysis** — Paste text, fill a structured form, or upload a screenshot
- **Google Gemini AI** — Contextual scam reasoning with multimodal vision support
- **Local ML Classifier** — TF-IDF + Logistic Regression for a supervised baseline
- **Rule-Based Detection** — Deterministic patterns for financial red flags, urgency tactics, suspicious URLs, and sensitive data requests
- **Risk Aggregation** — Configurable weighted combination of all three signals
- **Graceful degradation** — Works with any combination of available providers
- **Developer friendly** — Swap datasets, models, and providers without touching UI code

---

## Quick Start

### 1. Install Dependencies

```bash
cd scamcheck
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and set your Gemini API key:
# GEMINI_API_KEY=your-api-key-here
```

### 3. Train the Local ML Model (Optional)

```bash
python scripts/train_model.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Project Structure

```
scamcheck/
├── app.py                    # Main Streamlit application
├── config.yaml               # Central configuration
├── requirements.txt
├── .env.example
│
├── backend/
│   ├── config.py             # Config loader
│   ├── schemas.py            # Internal data schemas
│   ├── risk_engine.py        # Risk aggregation logic
│   ├── url_analyzer.py       # URL heuristic analysis
│   │
│   ├── models/
│   │   ├── base.py           # Abstract provider interface
│   │   ├── gemini_provider.py
│   │   └── local_ml_provider.py
│   │
│   ├── prompts/
│   │   └── scam_analysis_prompt.py   # All Gemini prompts
│   │
│   ├── rules/                # Deterministic rule detectors
│   │   ├── financial.py
│   │   ├── urgency.py
│   │   ├── links.py
│   │   ├── employment.py
│   │   └── sensitive_data.py
│   │
│   ├── services/
│   │   ├── analysis_service.py   # Main orchestration service
│   │   └── dataset_service.py    # Dataset loading & normalization
│   │
│   └── input_processors/
│       ├── text_processor.py
│       └── image_processor.py
│
├── scripts/
│   ├── train_model.py        # ML training pipeline
│   └── evaluate_model.py     # Model evaluation
│
├── data/
│   ├── raw/                  # Place your CSV dataset here
│   └── demo/                 # Built-in demo examples
│
├── models/                   # Trained model artifacts (auto-generated)
│
└── tests/
    ├── test_rules.py
    ├── test_risk_engine.py
    ├── test_schemas.py
    └── test_preprocessing.py
```

---

## Configuration

### Changing the Dataset

Edit `config.yaml`:

```yaml
dataset:
  path: data/raw/your_dataset.csv
  text_column: message     # column with opportunity text
  label_column: label      # column with scam/legitimate labels
  labels:
    scam: [scam, spam, fraud, "1"]
    legitimate: [legitimate, ham, safe, "0"]
```

Then retrain:

```bash
python scripts/train_model.py
```

### Changing the Gemini Model

In `.env`:

```
GEMINI_MODEL=gemini-2.0-flash
```

Or in `config.yaml` under `gemini.model`.

### Adjusting Risk Weights

In `config.yaml`:

```yaml
risk_engine:
  ml_weight: 0.30
  gemini_weight: 0.50
  rules_weight: 0.20
```

### Adding a New Provider

Create a new class extending `ScamAnalysisProvider` in `backend/models/`:

```python
from backend.models.base import ScamAnalysisProvider
from backend.schemas import AnalysisResult

class MyProvider(ScamAnalysisProvider):
    def analyze_text(self, text: str) -> AnalysisResult:
        ...
```

Register it in `backend/services/analysis_service.py`. The frontend requires no changes.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Risk Score Guide

| Score | Level | Verdict |
|-------|-------|---------|
| 0–24 | 🟢 LOW | Likely Legitimate |
| 25–49 | 🟡 MEDIUM | Suspicious |
| 50–74 | 🟠 HIGH | Potential Scam |
| 75–100 | 🔴 CRITICAL | High Risk Scam |

---

## Important Disclaimer

ScamCheck is an AI-powered **decision support tool**. It identifies recruitment scam indicators and produces an interpretable risk score. It does **not** guarantee fraud detection. Always verify opportunities independently through official company channels before taking any action.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| AI Provider | Google Gemini API |
| Local ML | scikit-learn (TF-IDF + Logistic Regression) |
| Language | Python 3.10+ |
| Config | YAML + python-dotenv |
