"""Structured schema for the Contract Risk Reviewer.

These Pydantic models define the contract-review "output contract": every clause
the agent extracts and every risk it flags is validated against these models, so
the report is always well-formed (typed severity, citations, categories) instead
of free text. The LLM is asked to emit JSON matching `LLMExtraction`; a rule-based
fallback populates the same models when no API key is available.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - pydantic is a hard dependency, but degrade gracefully
    PYDANTIC_AVAILABLE = False
    raise


class Severity(str, Enum):
    """Ordered risk severity. `order` is used for sorting / summary rollups."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

    @property
    def order(self) -> int:
        return {"High": 0, "Medium": 1, "Low": 2, "Info": 3}[self.value]


class RiskFinding(BaseModel):
    """A single flagged issue in the contract."""

    category: str = Field(..., description="Risk category, e.g. 'Auto-renewal', 'Unlimited liability'.")
    severity: Severity = Field(..., description="How dangerous this clause is for the reviewing party.")
    title: str = Field(..., description="Short human-readable label for the finding.")
    rationale: str = Field(..., description="Why this clause is risky or non-standard.")
    recommendation: str = Field("", description="Suggested redline / negotiation ask.")
    quote: str = Field(..., description="The exact contract text this finding is based on (verbatim citation).")
    clause_ref: Optional[str] = Field(None, description="Section/clause heading or number, if identifiable.")
    page: Optional[int] = Field(None, description="Source page number, if known (from the PDF).")
    source: str = Field("rule", description="'llm' or 'rule' — how the finding was produced.")

    def locator(self) -> str:
        """Human-readable 'p. 4 · §7' style anchor for fast verification."""
        parts = []
        if self.page:
            parts.append(f"p. {self.page}")
        if self.clause_ref:
            parts.append(self.clause_ref)
        return " · ".join(parts)

    def severity_order(self) -> int:
        return self.severity.order


class Clause(BaseModel):
    """A logical clause/section extracted from the contract."""

    heading: str = Field("", description="Detected heading or clause number.")
    text: str = Field(..., description="Full clause text.")
    start_char: int = Field(0, description="Character offset in the source document.")
    page: Optional[int] = Field(None, description="Source page number, if known (from the PDF).")


class RiskReport(BaseModel):
    """The full review deliverable — this is what the UI and eval consume."""

    contract_name: str = Field(..., description="Source file name.")
    overall_risk: Severity = Field(Severity.INFO, description="Highest-severity finding rolled up.")
    summary: str = Field("", description="Plain-English executive summary of the contract's risk posture.")
    findings: List[RiskFinding] = Field(default_factory=list)
    clause_count: int = Field(0, description="Number of clauses analyzed.")
    engine: str = Field("rule", description="'llm+rule' or 'rule' — which pipeline produced the report.")

    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    def rollup_overall(self) -> Severity:
        """Overall risk = worst finding severity (defaults to Info if clean)."""
        if not self.findings:
            return Severity.INFO
        return min((f.severity for f in self.findings), key=lambda s: s.order)

    def sorted_findings(self) -> List[RiskFinding]:
        return sorted(self.findings, key=lambda f: (f.severity_order(), f.category))


class LLMExtraction(BaseModel):
    """Shape the LLM is asked to return (validated before we trust it)."""

    findings: List[RiskFinding] = Field(default_factory=list)
    summary: str = Field("")
