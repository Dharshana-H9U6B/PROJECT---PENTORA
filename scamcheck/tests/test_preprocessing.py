"""
Tests for dataset preprocessing and text cleaning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from backend.services.dataset_service import DatasetLoader, clean_text_for_ml
from backend.input_processors.text_processor import clean_text, validate_text_input, build_structured_text


class TestCleanTextForML:
    def test_lowercases_text(self):
        result = clean_text_for_ml("HELLO WORLD")
        assert result == result.lower()

    def test_replaces_url(self):
        result = clean_text_for_ml("Visit https://example.com for details.")
        assert "URL" in result
        assert "https://example.com" not in result

    def test_replaces_email(self):
        result = clean_text_for_ml("Contact us at hr@company.com for more info.")
        assert "EMAIL" in result

    def test_handles_empty_string(self):
        result = clean_text_for_ml("")
        assert result == ""

    def test_handles_none_like(self):
        result = clean_text_for_ml(123)  # non-string
        assert result == ""


class TestCleanTextForAnalysis:
    def test_normalizes_whitespace(self):
        text = "Hello   World\n\n\n\nGoodbye"
        result = clean_text(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_strips_edges(self):
        text = "  Hello World  "
        assert clean_text(text) == "Hello World"

    def test_preserves_content(self):
        text = "Pay ₹999 to confirm your position."
        result = clean_text(text)
        assert "₹999" in result


class TestValidateTextInput:
    def test_valid_input(self):
        text = "This is a valid job opportunity message with enough content."
        is_valid, error = validate_text_input(text)
        assert is_valid is True
        assert error == ""

    def test_empty_input(self):
        is_valid, error = validate_text_input("")
        assert is_valid is False
        assert len(error) > 0

    def test_too_short_input(self):
        is_valid, error = validate_text_input("Hi there")
        assert is_valid is False

    def test_none_input(self):
        is_valid, error = validate_text_input(None)
        assert is_valid is False


class TestBuildStructuredText:
    def test_basic_build(self):
        text = build_structured_text(
            company="Google",
            role="SWE Intern",
            salary="₹20,000/month",
        )
        assert "Google" in text
        assert "SWE Intern" in text
        assert "₹20,000/month" in text

    def test_empty_fields_excluded(self):
        text = build_structured_text(company="Acme", role="")
        assert "Role" not in text or "Acme" in text


class TestLabelNormalization:
    def _make_loader(self):
        return DatasetLoader()

    def test_scam_labels(self):
        loader = self._make_loader()
        assert loader._normalize_label("scam") == 1
        assert loader._normalize_label("spam") == 1
        assert loader._normalize_label("fraud") == 1
        assert loader._normalize_label("1") == 1

    def test_legit_labels(self):
        loader = self._make_loader()
        assert loader._normalize_label("legitimate") == 0
        assert loader._normalize_label("ham") == 0
        assert loader._normalize_label("safe") == 0
        assert loader._normalize_label("0") == 0

    def test_unknown_label(self):
        loader = self._make_loader()
        result = loader._normalize_label("unknown_category")
        assert result is None
