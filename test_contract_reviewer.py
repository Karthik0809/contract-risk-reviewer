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


def test_page_anchor_from_markers():
    # DocuQuery-style page markers should populate the finding's page number.
    text = (
        "--- Page 1 ---\nGeneral provisions apply.\n\n"
        "--- Page 2 ---\nThe Contractor's liability is unlimited.\n"
    )
    report = review_contract(text, contract_name="paged.pdf")
    liab = [f for f in report.findings if f.category == "Unlimited liability"]
    assert liab and liab[0].page == 2
    assert "p. 2" in liab[0].locator()


def test_negation_not_flagged():
    # "will not automatically renew" must NOT trigger the auto-renewal rule.
    report = review_contract("This Agreement will not automatically renew.", contract_name="x")
    assert all(f.category != "Auto-renewal" for f in report.findings)


def test_pdf_and_docx_export():
    from report_export import report_to_docx, report_to_pdf
    report = review_contract(
        "Contractor's liability is unlimited and the Client may terminate for convenience.",
        contract_name="export_test",
    )
    pdf = report_to_pdf(report)
    docx = report_to_docx(report)
    assert pdf[:4] == b"%PDF"
    assert docx[:2] == b"PK"  # docx is a zip container
