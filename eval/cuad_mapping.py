"""Mapping between CUAD clause categories and Contract Risk Reviewer categories.

CUAD (Contract Understanding Atticus Dataset) is 510 real commercial contracts
with expert clause annotations across 41 categories. We evaluate against it to
get an *honest* recall number on real paper — not on clauses we wrote ourselves.

Only a subset of our categories has a faithful CUAD counterpart. We evaluate that
subset and are explicit about what CUAD can't score:

  Our category            CUAD category(ies)              Match quality
  ----------------------  ------------------------------  -------------
  Auto-renewal            Renewal Term                    good
  Unlimited liability     Uncapped Liability              excellent
  Unilateral termination  Termination For Convenience     excellent
  IP assignment           Ip Ownership Assignment         excellent
  Non-compete             Non-Compete                     good
  Governing law / venue   Governing Law                   partial (CUAD labels
                                                          any governing law; we
                                                          only flag *exclusive*
                                                          venue — expect lower
                                                          recall, and that's fair)

Not scorable against CUAD (no faithful category — reported, not counted):
  Indemnification, Payment terms, Confidentiality, Warranty disclaimer.

CUAD is licensed CC BY 4.0 (The Atticus Project). We never redistribute it — the
harness downloads it into a gitignored cache on demand.
"""

# our category -> list of CUAD category names (as they appear in CUAD questions)
CUAD_TO_OURS = {
    "Renewal Term": "Auto-renewal",
    "Uncapped Liability": "Unlimited liability",
    "Termination For Convenience": "Unilateral termination",
    "Ip Ownership Assignment": "IP assignment",
    "Non-Compete": "Non-compete",
    "Governing Law": "Governing law / venue",
}

# Our categories that CUAD cannot fairly score (kept out of the recall denominator).
UNSCORABLE = ["Indemnification", "Payment terms", "Confidentiality", "Warranty disclaimer"]

# Match-quality note per scorable category, surfaced in the report.
MATCH_QUALITY = {
    "Auto-renewal": "good",
    "Unlimited liability": "excellent",
    "Unilateral termination": "excellent",
    "IP assignment": "excellent",
    "Non-compete": "good",
    "Governing law / venue": "partial (we only flag exclusive venue)",
}

SCORABLE_OURS = sorted(set(CUAD_TO_OURS.values()))
