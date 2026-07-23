"""Smoke tests for the Contract Risk Reviewer (offline, no API key required).

Run: python -m pytest test_contract_reviewer.py -q
"""

from contract_agent import extract_clauses, review_contract
from contract_schema import RiskReport, Severity
from risk_rules import scan_text


def test_rule_scan_detects_auto_renewal():
    findings = scan_text("This Agreement shall automatically renew for successive one-year periods.")
    cats = {f.category for f in findings}
    assert "Auto-renewal" in cats


def test_clean_clause_has_no_findings():
    findings = scan_text("The parties shall cooperate in good faith and act reasonably.")
    assert findings == []


def test_extract_clauses_splits_headings():
    text = (
        "1. TERM\nThe term is 12 months.\n\n"
        "2. LIABILITY\nLiability is unlimited.\n\n"
        "3. GOVERNING LAW\nGoverned by the laws of Delaware.\n"
    )
    clauses = extract_clauses(text)
    assert len(clauses) >= 3


def test_review_contract_returns_validated_report():
    text = (
        "The Contractor's liability is unlimited. "
        "Vendor shall indemnify and hold harmless the Company. "
        "This Agreement shall automatically renew."
    )
    report = review_contract(text, contract_name="unit_test.txt")
    assert isinstance(report, RiskReport)
    assert report.overall_risk == Severity.HIGH
    cats = {f.category for f in report.findings}
    assert {"Unlimited liability", "Indemnification", "Auto-renewal"} <= cats
    # every finding carries a verbatim citation
    assert all(f.quote for f in report.findings)


def test_report_json_roundtrips():
    report = review_contract("Payment terms are net 90.", contract_name="x")
    payload = report.model_dump_json()
    assert "Payment terms" in payload
