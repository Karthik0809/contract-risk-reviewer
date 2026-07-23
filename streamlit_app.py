"""Contract Risk Reviewer — Streamlit front end.

Upload a contract (PDF or text) -> the agent analyzes it -> a color-coded risk
report with severity badges, an executive summary, and expandable verbatim
citations for every flagged clause.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from contract_agent import review_contract
from contract_schema import RiskReport, Severity

# Reuse the bundled PDF extractor when available.
try:
    from document_processor import EnhancedDocumentProcessor
    _PROCESSOR = EnhancedDocumentProcessor()
except Exception:  # pragma: no cover
    _PROCESSOR = None

# Reuse the bundled Gemini wrapper for the optional LLM pass.
try:
    from llm_interface import GeminiGenerator
except Exception:  # pragma: no cover
    GeminiGenerator = None


SEVERITY_COLORS = {
    Severity.HIGH: "#d64545",
    Severity.MEDIUM: "#e08e0b",
    Severity.LOW: "#c9a227",
    Severity.INFO: "#4a7fb5",
}


def _badge(severity: Severity) -> str:
    color = SEVERITY_COLORS.get(severity, "#777")
    return (
        f"<span style='background:{color};color:#fff;padding:2px 10px;"
        f"border-radius:12px;font-size:0.8rem;font-weight:600'>{severity.value}</span>"
    )


def _extract_text(uploaded) -> tuple[str, str]:
    name = uploaded.name
    suffix = os.path.splitext(name)[1].lower()
    if suffix == ".pdf" and _PROCESSOR is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name
        try:
            text, _meta = _PROCESSOR.extract_text(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return text, name
    # txt / md / fallback
    raw = uploaded.getvalue()
    try:
        return raw.decode("utf-8", errors="ignore"), name
    except Exception:
        return str(raw), name


@st.cache_resource(show_spinner=False)
def _get_generator(api_key: str):
    if not api_key or GeminiGenerator is None:
        return None
    gen = GeminiGenerator()
    ok = gen.configure(api_key)
    return gen if ok else None


def render_report(report: RiskReport) -> None:
    overall = report.overall_risk
    st.markdown(
        f"### Overall risk: {_badge(overall)}  &nbsp; "
        f"<span style='color:#888'>({report.engine} engine · {report.clause_count} clauses)</span>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total findings", len(report.findings))
    c2.metric("High severity", report.high_count())
    c3.metric("Medium severity", report.medium_count())

    st.markdown("#### Executive summary")
    st.write(report.summary)

    st.markdown("#### Flagged clauses")
    if not report.findings:
        st.success("No risky or non-standard clauses detected. Still review manually before signing.")
        return

    for i, f in enumerate(report.sorted_findings(), start=1):
        header = f"{_badge(f.severity)} &nbsp; **{f.category}** — {f.title}"
        with st.expander(f"{i}. [{f.severity.value}] {f.category} — {f.title}", expanded=(f.severity == Severity.HIGH)):
            st.markdown(header, unsafe_allow_html=True)
            st.markdown(f"**Why it's risky:** {f.rationale}")
            if f.recommendation:
                st.markdown(f"**Recommended action:** {f.recommendation}")
            if f.clause_ref:
                st.caption(f"Clause reference: {f.clause_ref}")
            st.markdown("**Cited text:**")
            st.markdown(
                f"<blockquote style='border-left:3px solid {SEVERITY_COLORS.get(f.severity)};"
                f"padding-left:12px;color:#444;font-style:italic'>{f.quote}</blockquote>",
                unsafe_allow_html=True,
            )
            st.caption(f"Source: {f.source}")


def main() -> None:
    st.set_page_config(page_title="Contract Risk Reviewer", page_icon="📑", layout="wide")
    st.title("📑 Contract Risk Reviewer")
    st.caption(
        "Upload a contract and get a risk report in seconds — flagged risky/non-standard "
        "clauses with severity and exact citations. Powered by a LangGraph agent over a hybrid-retrieval core."
    )

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Google Gemini API key (optional)",
            type="password",
            value=os.environ.get("DEFAULT_GEMINI_API_KEY", ""),
            help="Without a key the reviewer still runs the deterministic rule engine offline.",
        )
        st.markdown(
            "- **With key:** LLM + rule engine (catches non-standard clauses too)\n"
            "- **Without key:** rule engine only (fully offline)"
        )
        st.divider()
        st.markdown("**Detected risk categories**")
        from risk_rules import CATEGORIES
        st.caption(", ".join(CATEGORIES))

    uploaded = st.file_uploader("Upload a contract", type=["pdf", "txt", "md"])
    demo = st.checkbox("Use built-in sample contract", value=not uploaded)

    text, name = "", ""
    if uploaded is not None:
        text, name = _extract_text(uploaded)
    elif demo:
        sample_path = os.path.join(os.path.dirname(__file__), "eval", "samples", "sample_risky_contract.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            name = "sample_risky_contract.txt"

    if not text.strip():
        st.info("Upload a PDF/TXT contract or tick the sample checkbox to see a live report.")
        return

    if st.button("Analyze contract", type="primary") or demo or uploaded:
        with st.spinner("Analyzing contract…"):
            generator = _get_generator(api_key)
            report = review_contract(text, contract_name=name, generator=generator)
        render_report(report)

        st.download_button(
            "Download report (JSON)",
            data=report.model_dump_json(indent=2),
            file_name=f"{os.path.splitext(name)[0]}_risk_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
