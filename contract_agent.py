"""LangGraph agent that turns a contract into a structured RiskReport.

Pipeline (graph nodes):

    extract_clauses  ->  rule_check  ->  llm_check  ->  build_report

- extract_clauses: split raw text into logical clauses (headings/numbered sections).
- rule_check:      deterministic regex catalog (risk_rules) — always runs, offline.
- llm_check:       optional Gemini structured extraction validated against the
                   Pydantic schema; catches non-standard clauses the rules miss.
- build_report:    merge + de-duplicate findings, roll up overall severity, write
                   an executive summary.

The graph is built with LangGraph when available; otherwise the same nodes run in
a plain linear fallback, so the agent works even without the langgraph package.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, TypedDict

from contract_schema import Clause, LLMExtraction, RiskFinding, RiskReport, Severity
from risk_rules import CATEGORIES, scan_text

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Clause extraction
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(
    r"(?m)^\s*(?:"
    r"(?:section|clause|article)\s+\d+[\.\):]?"      # "Section 3."
    r"|\d+(?:\.\d+)*[\.\)]\s+[A-Z]"                  # "3.1 Confidentiality"
    r"|[A-Z][A-Z \-/]{3,}$"                          # "LIMITATION OF LIABILITY"
    r")"
)


_PAGE_MARKER_RE = re.compile(r"---\s*Page\s+(\d+)\s*---")


def _build_page_index(text: str) -> List[tuple[int, int]]:
    """Return sorted (char_offset, page_number) breakpoints from '--- Page N ---' markers.

    The DocuQuery PDF extractor inserts these markers; TXT inputs have none, in
    which case this returns [] and page lookups yield None.
    """
    return [(m.start(), int(m.group(1))) for m in _PAGE_MARKER_RE.finditer(text)]


def _page_for(offset: int, index: List[tuple[int, int]]) -> Optional[int]:
    if not index:
        return None
    page = None
    for pos, num in index:
        if pos <= offset:
            page = num
        else:
            break
    return page


def extract_clauses(text: str) -> List[Clause]:
    """Split a contract into clauses by detected headings; fall back to paragraphs."""
    text = text or ""
    page_index = _build_page_index(text)
    heads = list(_HEADING_RE.finditer(text))
    clauses: List[Clause] = []

    if len(heads) >= 3:
        for i, h in enumerate(heads):
            start = h.start()
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[start:end].strip()
            if len(body) < 15:
                continue
            # Use the clause's first non-empty line as a readable heading label
            # (e.g. "1. TERM AND RENEWAL"), skipping any "--- Page N ---" marker.
            heading = ""
            for line in body.splitlines():
                line = line.strip()
                if line and not _PAGE_MARKER_RE.fullmatch(line):
                    heading = re.sub(r"\s+", " ", line)[:120]
                    break
            clauses.append(Clause(heading=heading, text=body, start_char=start, page=_page_for(start, page_index)))
    else:
        # No clear structure: chunk on blank lines / paragraph boundaries.
        offset = 0
        for para in re.split(r"\n\s*\n", text):
            body = para.strip()
            if len(body) >= 40:
                clauses.append(Clause(heading="", text=body, start_char=offset, page=_page_for(offset, page_index)))
            offset += len(para) + 2

    if not clauses and text.strip():
        clauses.append(Clause(heading="", text=text.strip(), start_char=0, page=_page_for(0, page_index)))
    return clauses


# --------------------------------------------------------------------------- #
# LLM structured extraction (optional)
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You are a contract risk analyst. Review the contract text and identify clauses that are "
    "risky or non-standard for the reviewing party (assume the reviewer is a small business or "
    "freelancer, NOT the drafting party).\n\n"
    "Return ONLY valid JSON with this exact shape:\n"
    '{\n'
    '  "summary": "2-4 sentence plain-English risk summary",\n'
    '  "findings": [\n'
    '    {\n'
    '      "category": "<one of the categories below, or a short new one>",\n'
    '      "severity": "High" | "Medium" | "Low" | "Info",\n'
    '      "title": "short label",\n'
    '      "rationale": "why this is risky",\n'
    '      "recommendation": "suggested redline",\n'
    '      "quote": "EXACT verbatim text copied from the contract",\n'
    '      "clause_ref": "section heading/number or null"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "Rules: the 'quote' MUST be copied verbatim from the contract (used as a citation). "
    "Do not invent clauses. If the contract is clean, return an empty findings list.\n"
    "Preferred categories: " + ", ".join(CATEGORIES) + "."
)


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # grab the outermost object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return None


def llm_extract(text: str, generator) -> Optional[LLMExtraction]:
    """Run Gemini structured extraction. `generator` is a configured GeminiGenerator.

    Returns None if the LLM is unavailable or the output can't be validated.
    """
    if generator is None or not getattr(generator, "api_key_set", False) or not getattr(generator, "model", None):
        return None
    # Keep the prompt within a safe size; contracts are chunked upstream if huge.
    body = text[:18000]
    prompt = f"{_LLM_SYSTEM}\n\n---- CONTRACT START ----\n{body}\n---- CONTRACT END ----\n\nJSON:"
    try:
        resp = generator.model.generate_content(prompt)
        data = _extract_json(getattr(resp, "text", "") or "")
        if data is None:
            return None
        # Validate findings item-by-item and normalize common LLM quirks (lowercase
        # severity, missing optional fields) so one malformed item doesn't discard
        # the whole extraction.
        valid: List[RiskFinding] = []
        for f in (data.get("findings") or []):
            try:
                f = dict(f)
                f["source"] = "llm"
                sev = str(f.get("severity", "")).strip().lower()
                f["severity"] = {"high": "High", "medium": "Medium",
                                 "low": "Low", "info": "Info"}.get(sev, "Medium")
                if not (f.get("quote") or "").strip():
                    continue  # a finding without a citation isn't usable
                f.setdefault("category", "Other")
                f.setdefault("title", f.get("category", "Issue"))
                f.setdefault("rationale", "")
                if f.get("clause_ref") in ("", "null", None):
                    f["clause_ref"] = None
                valid.append(RiskFinding.model_validate(f))
            except Exception:  # noqa: BLE001 - skip a single bad item, keep the rest
                continue
        return LLMExtraction(findings=valid, summary=(data.get("summary") or "").strip())
    except Exception as e:  # noqa: BLE001 - LLM failures must never crash the report
        print(f"[contract_agent] LLM extraction failed, using rules only: {e}")
        return None


# --------------------------------------------------------------------------- #
# Merge / de-duplicate
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def merge_findings(rule_findings: List[RiskFinding], llm_findings: List[RiskFinding]) -> List[RiskFinding]:
    """Union rule + LLM findings, dropping near-duplicates (same category + overlapping quote)."""
    merged: List[RiskFinding] = list(rule_findings)
    for lf in llm_findings:
        dup = False
        lq = _norm(lf.quote)[:80]
        for existing in merged:
            if existing.category == lf.category and lq and lq[:40] in _norm(existing.quote):
                dup = True
                break
        if not dup:
            merged.append(lf)
    return merged


def _fallback_summary(report_name: str, findings: List[RiskFinding]) -> str:
    if not findings:
        return f"No risky or non-standard clauses were detected in {report_name}. Review manually to confirm."
    highs = [f for f in findings if f.severity == Severity.HIGH]
    cats = sorted({f.category for f in findings})
    lead = (
        f"{len(findings)} potential issue(s) flagged in {report_name}"
        + (f", including {len(highs)} high-severity" if highs else "")
        + f". Categories: {', '.join(cats)}."
    )
    if highs:
        lead += " Prioritize the high-severity items before signing."
    return lead


# --------------------------------------------------------------------------- #
# Agent state + nodes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    text: str
    contract_name: str
    generator: object
    clauses: List[Clause]
    rule_findings: List[RiskFinding]
    llm_findings: List[RiskFinding]
    llm_summary: str
    report: RiskReport


def _node_extract(state: AgentState) -> AgentState:
    state["clauses"] = extract_clauses(state.get("text", ""))
    return state


def _node_rules(state: AgentState) -> AgentState:
    findings: List[RiskFinding] = []
    for c in state.get("clauses", []):
        findings.extend(scan_text(c.text, clause_ref=(c.heading or None), page=c.page))
    state["rule_findings"] = findings
    return state


def _node_llm(state: AgentState) -> AgentState:
    extraction = llm_extract(state.get("text", ""), state.get("generator"))
    if extraction:
        state["llm_findings"] = extraction.findings
        state["llm_summary"] = extraction.summary
    else:
        state["llm_findings"] = []
        state["llm_summary"] = ""
    return state


def _node_report(state: AgentState) -> AgentState:
    name = state.get("contract_name", "contract")
    text = state.get("text", "")
    page_index = _build_page_index(text)
    # Backfill page numbers for LLM findings by locating their verbatim quote.
    for f in state.get("llm_findings", []):
        if f.page is None and f.quote and page_index:
            idx = text.find(f.quote[:60])
            if idx != -1:
                f.page = _page_for(idx, page_index)
    merged = merge_findings(state.get("rule_findings", []), state.get("llm_findings", []))
    used_llm = bool(state.get("llm_findings")) or bool(state.get("llm_summary"))
    summary = state.get("llm_summary") or _fallback_summary(name, merged)
    report = RiskReport(
        contract_name=name,
        summary=summary,
        findings=merged,
        clause_count=len(state.get("clauses", [])),
        engine="llm+rule" if used_llm else "rule",
    )
    report.overall_risk = report.rollup_overall()
    state["report"] = report
    return state


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_clauses", _node_extract)
    graph.add_node("rule_check", _node_rules)
    graph.add_node("llm_check", _node_llm)
    graph.add_node("build_report", _node_report)
    graph.set_entry_point("extract_clauses")
    graph.add_edge("extract_clauses", "rule_check")
    graph.add_edge("rule_check", "llm_check")
    graph.add_edge("llm_check", "build_report")
    graph.add_edge("build_report", END)
    return graph.compile()


_COMPILED = _build_graph() if LANGGRAPH_AVAILABLE else None


def review_contract(text: str, contract_name: str = "contract", generator=None) -> RiskReport:
    """Public entry point: run the agent and return a validated RiskReport.

    Uses the LangGraph pipeline when available, else an equivalent linear run.
    `generator` is an optional configured GeminiGenerator (from llm_interface).
    """
    state: AgentState = {"text": text or "", "contract_name": contract_name, "generator": generator}
    if _COMPILED is not None:
        result = _COMPILED.invoke(state)
        return result["report"]
    # Linear fallback (no langgraph installed)
    for node in (_node_extract, _node_rules, _node_llm, _node_report):
        state = node(state)
    return state["report"]
