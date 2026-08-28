"""Evaluate the Contract Risk Reviewer against CUAD — 510 real commercial contracts.

For each contract, CUAD tells us which clause categories are actually present
(expert-labeled). We run our reviewer on the full contract text and measure, for
the mappable subset of categories, how many present clauses we detect (recall).

Why recall and not precision here: CUAD only annotates its own 41 categories, so a
flag we raise for a category CUAD doesn't track (e.g. Indemnification) can't be
called a false positive. Precision is measured on our own labeled set with hard
negatives (`run_eval.py`); CUAD is the real-world *recall* check.

Data: CUAD ships a SQuAD-format `data.zip` (JSON only, ~18 MB) on GitHub. The
harness downloads it on demand into a gitignored cache and parses it directly —
no heavy `datasets` dependency. CUAD is CC BY 4.0 (The Atticus Project) and is
never redistributed by this repo.

Usage:
    python eval/run_cuad_eval.py                   # rules-only, all test contracts
    python eval/run_cuad_eval.py --limit 30        # quick subset
    python eval/run_cuad_eval.py --split full      # all 510 contracts (CUADv1.json)
    python eval/run_cuad_eval.py --llm --limit 20  # include Gemini pass (costs API calls)
    python eval/run_cuad_eval.py --save            # write eval/cuad_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract_agent import review_contract  # noqa: E402
from eval.cuad_mapping import (  # noqa: E402
    CUAD_TO_OURS,
    MATCH_QUALITY,
    SCORABLE_OURS,
    UNSCORABLE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cuad_cache")
RESULTS_PATH = os.path.join(HERE, "cuad_results.json")
DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

# CUAD question text looks like: ... related to "Renewal Term" that should be ...
_CAT_RE = re.compile(r'related to\s+"([^"]+)"')


def _ensure_data() -> str:
    """Download CUAD data.zip into the gitignored cache if absent; return its path."""
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, "data.zip")
    if not os.path.exists(dst):
        print(f"Downloading CUAD data.zip (~18 MB) from {DATA_URL} …")
        urllib.request.urlretrieve(DATA_URL, dst)
    return dst


def _load_cuad(split: str):
    """Return the SQuAD-format 'data' list. split: 'test' (102) or 'full' (510)."""
    member = "test.json" if split == "test" else "CUADv1.json"
    with zipfile.ZipFile(_ensure_data()) as z:
        return json.loads(z.read(member))["data"]


def _category_from_question(q: str) -> str | None:
    m = _CAT_RE.search(q or "")
    return m.group(1).strip() if m else None


def _group_by_contract(data):
    """Aggregate SQuAD entries into {title: context} and {title: {present CUAD categories}}."""
    contexts: dict[str, str] = {}
    present: dict[str, set] = defaultdict(set)
    for entry in data:
        title = entry["title"]
        for para in entry["paragraphs"]:
            contexts.setdefault(title, para["context"])
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                if not qa.get("answers"):
                    continue
                cat = _category_from_question(qa["question"])
                if cat:
                    present[title].add(cat)
    return contexts, present


def evaluate(split: str = "test", limit: int | None = None, use_llm: bool = False) -> dict:
    data = _load_cuad(split)
    contexts, present = _group_by_contract(data)

    titles = list(contexts.keys())
    if limit:
        titles = titles[:limit]

    generator = None
    if use_llm:
        try:
            from llm_interface import GeminiGenerator
            key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("DEFAULT_GEMINI_API_KEY", "")
            if key:
                generator = GeminiGenerator()
                if not generator.configure(key):
                    generator = None
        except Exception as e:  # noqa: BLE001
            print(f"LLM unavailable ({e}); running rules-only.")

    per_cat = {c: {"tp": 0, "gold": 0} for c in SCORABLE_OURS}

    # Track whether the LLM pass actually ran. Without this the harness happily
    # labels a run "llm+rule" even when every call 429s and silently falls back to
    # rules -- which produces rules-only numbers wearing an LLM label.
    llm_attempted = 0
    llm_failed = 0

    for i, title in enumerate(titles, start=1):
        text = contexts[title]
        gold_ours = {CUAD_TO_OURS[c] for c in present.get(title, set()) if c in CUAD_TO_OURS}
        report = review_contract(text, contract_name=title, generator=generator)
        if generator is not None:
            llm_attempted += 1
            if getattr(report, "llm_error", None):
                llm_failed += 1
        predicted = {f.category for f in report.findings}
        for c in SCORABLE_OURS:
            if c in gold_ours:
                per_cat[c]["gold"] += 1
                if c in predicted:
                    per_cat[c]["tp"] += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(titles)} contracts")

    total_tp = sum(v["tp"] for v in per_cat.values())
    total_gold = sum(v["gold"] for v in per_cat.values())
    overall_recall = total_tp / total_gold if total_gold else 0.0

    # Label the engine by what actually ran, not by what was requested.
    llm_ok = llm_attempted - llm_failed
    if generator is None:
        engine = "rule"
    elif llm_ok == 0:
        engine = "rule (LLM requested but every call failed)"
    elif llm_failed:
        engine = f"llm+rule (degraded: {llm_failed}/{llm_attempted} LLM calls failed)"
    else:
        engine = "llm+rule"

    return {
        "dataset": "CUAD v1 (The Atticus Project, CC BY 4.0)",
        "split": split,
        "engine": engine,
        "llm_attempted": llm_attempted,
        "llm_failed": llm_failed,
        "n_contracts": len(titles),
        "overall_recall_scorable": round(overall_recall, 3),
        "scorable_gold_clauses": total_gold,
        "per_category": {
            c: {
                "recall": round(per_cat[c]["tp"] / per_cat[c]["gold"], 3) if per_cat[c]["gold"] else None,
                "gold": per_cat[c]["gold"],
                "match_quality": MATCH_QUALITY.get(c, ""),
            }
            for c in SCORABLE_OURS
        },
        "not_scorable_against_cuad": UNSCORABLE,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "full"],
                    help="'test' = 102 contracts; 'full' = all 510 (CUADv1.json).")
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate the first N contracts.")
    ap.add_argument("--llm", action="store_true", help="Include the Gemini pass (costs API calls).")
    ap.add_argument("--save", action="store_true", help="Write results to eval/cuad_results.json.")
    args = ap.parse_args()

    print(f"Loading CUAD ({args.split})… (first run downloads ~18 MB into eval/.cuad_cache/)")
    results = evaluate(split=args.split, limit=args.limit, use_llm=args.llm)

    print("\n=== Contract Risk Reviewer — CUAD real-world evaluation ===")
    print(f"Dataset:   {results['dataset']}")
    print(f"Split:     {results['split']}  |  contracts: {results['n_contracts']}  |  engine: {results['engine']}")
    if results.get("llm_failed"):
        print(
            f"\n!! WARNING: {results['llm_failed']} of {results['llm_attempted']} LLM calls failed "
            f"(commonly free-tier 429 quota).\n"
            f"!! Those contracts were scored by the rule engine alone, so the numbers below\n"
            f"!! UNDERSTATE the LLM pass. Do not report this as a clean llm+rule result."
        )
    print(f"\nOverall recall (scorable categories): {results['overall_recall_scorable']:.1%}  "
          f"(n={results['scorable_gold_clauses']} labeled clauses)")
    print("\nPer-category recall on real contracts:")
    print(f"  {'Category':<24}{'Recall':>8}{'Gold':>7}   Match quality")
    for c, v in results["per_category"].items():
        r = f"{v['recall']:.0%}" if v["recall"] is not None else "n/a"
        print(f"  {c:<24}{r:>8}{v['gold']:>7}   {v['match_quality']}")
    print(f"\nNot scorable against CUAD (no faithful category): {', '.join(results['not_scorable_against_cuad'])}")
    print(
        "\nReading this: the deterministic rules are precision-first, so on real,\n"
        "diversely-worded contracts they form a high-precision / lower-recall baseline.\n"
        "This is the number the optional LLM pass exists to lift — run with --llm to\n"
        "measure that lift on the same contracts."
    )
    print()

    if args.save:
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
