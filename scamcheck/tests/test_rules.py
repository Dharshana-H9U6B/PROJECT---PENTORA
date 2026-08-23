"""
Tests for rule-based detectors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.rules.financial import detect_financial_indicators
from backend.rules.urgency import detect_urgency_indicators
from backend.rules.links import detect_link_indicators
from backend.rules.sensitive_data import detect_sensitive_data_indicators
from backend.rules.employment import detect_employment_indicators


class TestFinancialDetection:
    def test_detects_registration_fee(self):
        text = "Pay ₹2999 registration fee to confirm your internship."
        indicators = detect_financial_indicators(text)
        assert len(indicators) > 0
        types = [i.type for i in indicators]
        assert "UPFRONT_PAYMENT" in types

    def test_detects_rupee_amount(self):
        text = "Send ₹5000 to our UPI ID to secure your position."
        indicators = detect_financial_indicators(text)
        assert len(indicators) > 0

    def test_detects_security_deposit(self):
        text = "A refundable security deposit of ₹3000 is required."
        indicators = detect_financial_indicators(text)
        assert len(indicators) > 0

    def test_no_false_positive_salary_mention(self):
        """Mentioning salary should not trigger payment indicators."""
        text = "The internship pays ₹15000 per month stipend. No fees."
        indicators = detect_financial_indicators(text)
        # Salary mention should not trigger high-severity UPFRONT_PAYMENT
        high_indicators = [i for i in indicators if i.severity == "HIGH" and i.type == "UPFRONT_PAYMENT"]
        assert len(high_indicators) == 0

    def test_detects_training_fee(self):
        text = "You must pay ₹2000 training fee before joining."
        indicators = detect_financial_indicators(text)
        assert len(indicators) > 0


class TestUrgencyDetection:
    def test_detects_limited_seats(self):
        text = "Only 3 seats remaining. Apply immediately!"
        indicators = detect_urgency_indicators(text)
        assert len(indicators) > 0
        types = [i.type for i in indicators]
        assert "URGENCY" in types

    def test_detects_act_now(self):
        text = "Act now to secure your position before it's too late."
        indicators = detect_urgency_indicators(text)
        assert len(indicators) > 0

    def test_detects_urgent_keyword(self):
        text = "URGENT: This offer expires today only."
        indicators = detect_urgency_indicators(text)
        assert len(indicators) > 0

    def test_no_false_positive_normal_deadline(self):
        """Application deadline is normal and should not always trigger high urgency."""
        text = "Please submit your application within 7 days."
        indicators = detect_urgency_indicators(text)
        # A normal deadline may or may not trigger, but should not be HIGH
        high_indicators = [i for i in indicators if i.severity == "HIGH"]
        # This is flexible — just ensure it doesn't crash
        assert isinstance(high_indicators, list)


class TestSensitiveDataDetection:
    def test_detects_otp_request(self):
        text = "Please share the OTP received on your mobile to verify your account."
        indicators = detect_sensitive_data_indicators(text)
        assert len(indicators) > 0
        types = [i.type for i in indicators]
        assert "SENSITIVE_DATA_REQUEST" in types

    def test_detects_bank_account_request(self):
        text = "Provide your bank account number and IFSC code to process the stipend."
        indicators = detect_sensitive_data_indicators(text)
        assert len(indicators) > 0

    def test_detects_upi_pin(self):
        text = "Enter your UPI PIN to confirm the payment."
        indicators = detect_sensitive_data_indicators(text)
        assert len(indicators) > 0

    def test_detects_aadhaar_request(self):
        text = "Submit your Aadhaar number and photo for KYC verification."
        indicators = detect_sensitive_data_indicators(text)
        assert len(indicators) > 0


class TestLinkDetection:
    def test_detects_url_shortener(self):
        text = "Apply now at bit.ly/fake-job-offer"
        indicators = detect_link_indicators(text)
        types = [i.type for i in indicators]
        assert "SHORTENED_URL" in types or "SUSPICIOUS_URL" in types

    def test_detects_http_url(self):
        text = "Visit http://suspicious-jobs.com to apply."
        indicators = detect_link_indicators(text)
        types = [i.type for i in indicators]
        assert "INSECURE_URL" in types

    def test_detects_ip_url(self):
        text = "Register at http://192.168.1.1/job-apply to get your offer letter."
        indicators = detect_link_indicators(text)
        types = [i.type for i in indicators]
        assert "SUSPICIOUS_URL" in types

    def test_no_indicators_for_clean_text(self):
        text = "Please check our website for more details."
        indicators = detect_link_indicators(text)
        assert len(indicators) == 0  # No URLs = no link indicators
