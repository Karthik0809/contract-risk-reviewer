"""Contract Risk Reviewer — Streamlit front end.

Upload a contract (PDF or text) -> the agent analyzes it -> a color-coded risk
report with severity badges, an executive summary, and expandable verbatim
citations for every flagged clause.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
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
    """Return (generator_or_None, status). status ∈ {'none','unavailable','invalid','ok'}.

    Cached per api_key so we authenticate once, not on every rerun.
    """
    if not api_key:
        return None, "none"
    if GeminiGenerator is None:
        return None, "unavailable"
    gen = GeminiGenerator()
    ok = gen.configure(api_key)
    return (gen, "ok") if ok else (None, "invalid")


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

    # Make the LLM pass's contribution explicit vs. the rule engine.
    if report.engine.startswith("llm"):
        llm_added = sum(1 for f in report.findings if f.source == "llm")
        if llm_added:
            st.info(f"🧠 LLM pass added **{llm_added}** finding(s) the rule engine missed.")
        else:
            st.info(
                "🧠 LLM pass active — it reviewed the contract and surfaced no clauses "
                "beyond the rule engine. On clear-cut contracts the LLM often just confirms "
                "the rules; its value shows on messier, non-standard wording."
            )

    st.markdown("#### Executive summary")
    st.write(report.summary)

    st.markdown("#### Flagged clauses")
    if not report.findings:
        st.success("No risky or non-standard clauses detected. Still review manually before signing.")
        return

    for i, f in enumerate(report.sorted_findings(), start=1):
        loc = f.locator()
        loc_label = f"  ·  {loc}" if loc else ""
        with st.expander(
            f"{i}. [{f.severity.value}] {f.category} — {f.title}{loc_label}",
            expanded=(f.severity == Severity.HIGH),
        ):
            st.markdown(f"{_badge(f.severity)} &nbsp; **{f.category}** — {f.title}", unsafe_allow_html=True)
            st.markdown(f"**Why it's risky:** {f.rationale}")
            if loc:
                st.caption(f"📍 Location: {loc}")

            # Redline view: current clause vs suggested change, side by side.
            left, right = st.columns(2)
            with left:
                st.markdown("**Current clause (as written)**")
                st.markdown(
                    f"<blockquote style='border-left:3px solid {SEVERITY_COLORS.get(f.severity)};"
                    f"padding-left:12px;color:#c0392b;font-style:italic'>{f.quote}</blockquote>",
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown("**Suggested change (redline)**")
                st.markdown(
                    f"<blockquote style='border-left:3px solid #2e8b57;"
                    f"padding-left:12px;color:#1e7d4f'>{f.recommendation or 'Review with counsel.'}</blockquote>",
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
        # Authenticate once and show the user exactly which engine is active.
        generator, gstatus = _get_generator(api_key)
        if gstatus == "ok":
            st.success("✅ Gemini connected — LLM + rule engine active.")
        elif gstatus == "invalid":
            st.error(
                "⚠️ Key not accepted. A Google AI Studio key starts with **`AIza…`** "
                "(get one at aistudio.google.com/apikey). Running rule engine only."
            )
        elif gstatus == "unavailable":
            st.warning("google-generativeai not installed — rule engine only.")
        else:
            st.info("No key set — running the offline rule engine.")
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

    # Only (re)analyze when the contract or the active engine changes — otherwise
    # every rerun (expanding a clause, clicking a download) would re-call Gemini
    # and make the app feel slow.
    engine_active = generator is not None
    sig = hashlib.sha1(f"{engine_active}|{name}|{text}".encode("utf-8")).hexdigest()
    if st.session_state.get("analysis_sig") != sig:
        spinner_msg = (
            "Analyzing with Gemini + rules… (runs once, then cached)"
            if engine_active else "Analyzing contract…"
        )
        with st.spinner(spinner_msg):
            report = review_contract(text, contract_name=name, generator=generator)
        st.session_state["analysis_sig"] = sig
        st.session_state["analysis_report"] = report
    report = st.session_state["analysis_report"]

    if engine_active and report.engine == "rule":
        msg = (
            "Gemini was connected but its analysis returned no usable output this run "
            "— showing rule-engine results."
        )
        if report.llm_error:
            msg += f"\n\n**Reason:** `{report.llm_error}`"
        else:
            msg += " This usually means an API quota/rate limit or a model-access issue."
        st.warning(msg)

    render_report(report)

    st.markdown("#### Export report")
    stem = os.path.splitext(name)[0]
    col_pdf, col_docx, col_json = st.columns(3)
    with col_pdf:
        try:
            from report_export import report_to_pdf
            st.download_button(
                "⬇ PDF", data=report_to_pdf(report),
                file_name=f"{stem}_risk_report.pdf", mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:  # noqa: BLE001
            st.caption(f"PDF unavailable: {e}")
    with col_docx:
        try:
            from report_export import report_to_docx
            st.download_button(
                "⬇ Word (DOCX)", data=report_to_docx(report),
                file_name=f"{stem}_risk_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as e:  # noqa: BLE001
            st.caption(f"DOCX unavailable: {e}")
    with col_json:
        st.download_button(
            "⬇ JSON", data=report.model_dump_json(indent=2),
            file_name=f"{stem}_risk_report.json", mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
