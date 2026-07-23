"""Rule catalog for contract risk detection.

Each rule pairs a risk category with regex patterns that signal a risky or
non-standard clause. Rules give the reviewer three things:

1. A deterministic, offline baseline that works with no API key.
2. A grounded citation (the matched span) for every flag.
3. A labeled taxonomy the LLM extractor is asked to align to, so LLM and rule
   findings de-duplicate cleanly.

Patterns are intentionally conservative (favor precision) but the eval harness
(`eval/run_eval.py`) measures recall against a labeled set so they can be tuned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern

from contract_schema import RiskFinding, Severity


@dataclass
class RiskRule:
    category: str
    severity: Severity
    title: str
    rationale: str
    recommendation: str
    patterns: List[str]
    _compiled: List[Pattern] = field(default_factory=list)

    def compile(self) -> "RiskRule":
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        return self

    def scan(self, text: str) -> List[re.Match]:
        matches = []
        for pat in self._compiled:
            matches.extend(pat.finditer(text))
        return matches


# --------------------------------------------------------------------------- #
# The catalog. Ordered roughly by how commonly they burn small businesses /
# freelancers reviewing a client's paper.
# --------------------------------------------------------------------------- #
_RULES: List[RiskRule] = [
    RiskRule(
        category="Auto-renewal",
        severity=Severity.HIGH,
        title="Automatic renewal / evergreen term",
        rationale="The contract renews itself unless you cancel within a narrow window, which can lock you in for another full term.",
        recommendation="Add an explicit non-renewal right and shorten the notice window, or convert to month-to-month after the initial term.",
        patterns=[
            r"(?<!not )automatically\s+renew\w*",
            r"auto[-\s]?renew\w*",
            r"evergreen",
            r"renew(?:s|ed|al)?\s+for\s+(?:successive|additional|further)\s+(?:period|term)",
            r"unless\s+(?:either\s+party|you|the\s+client)\s+(?:provides?|gives?)\s+(?:written\s+)?notice",
        ],
    ),
    RiskRule(
        category="Unlimited liability",
        severity=Severity.HIGH,
        title="Uncapped / unlimited liability",
        rationale="Liability is not capped, so damages could vastly exceed the contract value.",
        recommendation="Cap aggregate liability (commonly at fees paid in the prior 12 months) and exclude indirect/consequential damages.",
        patterns=[
            r"unlimited\s+liab\w*",
            r"liab\w*\s+(?:is|shall\s+be|will\s+be)\s+unlimited",
            r"without\s+(?:any\s+)?limit\w*\s+(?:of|on|to)\s+liab\w*",
            r"no\s+(?:cap|limit\w*)\s+on\s+[\w'’\s]{0,25}?liab\w*",
            r"liable\s+for\s+all\s+(?:damages|losses)",
        ],
    ),
    RiskRule(
        category="Indemnification",
        severity=Severity.HIGH,
        title="Broad indemnification obligation",
        rationale="You must cover the other party's losses, legal fees, or third-party claims — potentially open-ended.",
        recommendation="Make indemnities mutual, tie them to your breach/negligence, and cap them alongside the liability cap.",
        patterns=[
            r"indemnif\w+",
            r"hold\s+harmless",
            r"defend[,\s]+indemnif\w+",
        ],
    ),
    RiskRule(
        category="Unilateral termination",
        severity=Severity.MEDIUM,
        title="One-sided or convenience termination",
        rationale="The counterparty can terminate at will (or immediately) while you cannot, leaving your work or revenue exposed.",
        recommendation="Make termination-for-convenience mutual with equal notice, and require payment for work completed.",
        patterns=[
            r"terminate\s+(?:this\s+agreement\s+)?(?:at\s+any\s+time|for\s+convenience|without\s+cause)",
            r"(?:sole|its)\s+discretion[\w\s,]{0,25}?terminate",
            r"terminate\s+(?:this\s+agreement\s+)?immediately\s+without\s+notice",
        ],
    ),
    RiskRule(
        category="IP assignment",
        severity=Severity.MEDIUM,
        title="Broad intellectual property assignment",
        rationale="You may be assigning ownership of all work product, pre-existing IP, or tools — beyond the specific deliverables.",
        recommendation="Limit assignment to the deliverables; carve out your pre-existing IP and grant a license instead of ownership.",
        patterns=[
            r"assign\w*\s+all\s+(?:right|title|interest)",
            r"work\s+(?:made\s+)?for\s+hire",
            r"all\s+intellectual\s+property\b[\w\s,]{0,45}?(?:owned|assigned|assign\w*|vest)",
        ],
    ),
    RiskRule(
        category="Non-compete",
        severity=Severity.MEDIUM,
        title="Non-compete / exclusivity restriction",
        rationale="You may be barred from working with other clients or in a market for a period after the contract ends.",
        recommendation="Narrow the scope, geography, and duration, or remove it; ensure it does not block your normal line of business.",
        patterns=[
            r"non[-\s]?compet\w*",
            r"shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?(?:engage|compete|solicit)",
            r"exclusiv\w*\s+(?:to|with|arrangement)",
        ],
    ),
    RiskRule(
        category="Payment terms",
        severity=Severity.MEDIUM,
        title="Unfavorable payment terms",
        rationale="Long payment windows or unilateral fee changes hurt cash flow and pricing certainty.",
        recommendation="Shorten payment terms (e.g. Net 15–30), add late-payment interest, and require notice before any fee change.",
        patterns=[
            r"net\s+(?:60|90|120)",
            r"payment\s+within\s+(?:sixty|ninety|\d{2,3})\s+days",
            r"(?:may|reserves?\s+the\s+right\s+to)\s+(?:change|adjust|increase)\s+(?:the\s+)?(?:fees|prices|rates)",
        ],
    ),
    RiskRule(
        category="Governing law / venue",
        severity=Severity.LOW,
        title="Distant governing law or exclusive venue",
        rationale="Disputes must be litigated in a specific jurisdiction that may be inconvenient or costly for you.",
        recommendation="Negotiate a neutral or local governing law and venue, or agree to arbitration.",
        patterns=[
            # A plain governing-law clause is standard and not flagged; only an
            # *exclusive* venue/jurisdiction (which forecloses your options) is.
            r"exclusive\s+jurisdiction",
            r"exclusive\s+venue",
            r"venue\s+(?:shall|will)\s+(?:be|lie)\s+(?:exclusively\s+)?in",
            r"submit\s+to\s+the\s+exclusive\s+jurisdiction",
        ],
    ),
    RiskRule(
        category="Confidentiality",
        severity=Severity.LOW,
        title="Broad or perpetual confidentiality",
        rationale="Confidentiality obligations that are perpetual or one-sided can create long-tail compliance burden.",
        recommendation="Make confidentiality mutual and time-bound (e.g. 2–5 years post-termination) with standard carve-outs.",
        patterns=[
            r"in\s+perpetuity",
            r"perpetual\s+(?:confidentiality|obligation)",
            r"confidential\w*\s+(?:shall\s+)?surviv\w+\s+(?:indefinitely|in\s+perpetuity)",
        ],
    ),
    RiskRule(
        category="Warranty disclaimer",
        severity=Severity.LOW,
        title="Broad warranty disclaimer",
        rationale="The counterparty disclaims warranties, shifting quality/performance risk onto you.",
        recommendation="Preserve at least a limited warranty of conformance and a remedy for material defects.",
        patterns=[
            r"as[-\s]is",
            r"without\s+warrant\w*\s+of\s+any\s+kind",
            r"disclaim\w*\s+all\s+warrant\w*",
        ],
    ),
]

RULES: List[RiskRule] = [r.compile() for r in _RULES]

# Public taxonomy the LLM extractor is aligned to.
CATEGORIES = [r.category for r in RULES]


def _snippet(text: str, start: int, end: int, window: int = 200) -> str:
    """Return a readable verbatim citation: the sentence(s) containing the match.

    Snaps the start back to the nearest sentence/line boundary (so quotes never
    begin mid-word) and extends forward to the end of the matched sentence.
    """
    # snap start back to the previous sentence break or newline (within 240 chars)
    lo = start
    for i in range(start - 1, max(0, start - 240) - 1, -1):
        if text[i] in ".\n" and i + 1 < len(text) and text[i + 1] in " \n":
            lo = i + 1
            break
    else:
        lo = max(0, start - 240)
    # extend forward to the end of the sentence, capped by `window`
    hi = min(len(text), end + window)
    dot = text.find(".", end, hi)
    if dot != -1:
        hi = dot + 1
    snippet = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return snippet


def scan_text(text: str, clause_ref: str | None = None, page: int | None = None) -> List[RiskFinding]:
    """Run the full rule catalog over `text` and return findings.

    De-duplicates by (category, overlapping span) so the same clause matched by
    multiple patterns produces one finding.
    """
    findings: List[RiskFinding] = []
    seen_spans: list[tuple[str, int]] = []
    for rule in RULES:
        for m in rule.scan(text):
            # collapse near-duplicate hits of the same category
            if any(cat == rule.category and abs(pos - m.start()) < 120 for cat, pos in seen_spans):
                continue
            seen_spans.append((rule.category, m.start()))
            findings.append(
                RiskFinding(
                    category=rule.category,
                    severity=rule.severity,
                    title=rule.title,
                    rationale=rule.rationale,
                    recommendation=rule.recommendation,
                    quote=_snippet(text, m.start(), m.end()),
                    clause_ref=clause_ref,
                    page=page,
                    source="rule",
                )
            )
    return findings
