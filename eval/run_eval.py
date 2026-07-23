"""Evaluation harness for the Contract Risk Reviewer.

Measures how well the reviewer detects risky clauses against a labeled set
(`labeled_contracts.json`). Reports:

  - Detection rate (recall): of the risky clauses that SHOULD be flagged, how many were.
  - Precision: of the categories flagged, how many were correct.
  - Per-category recall.
  - Clean-clause false-positive rate.

Runs offline against the deterministic rule engine by default (no API key needed),
so results are reproducible. Pass --llm with GOOGLE_API_KEY set to also exercise
the LLM pass.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as `python eval/run_eval.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract_agent import review_contract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "labeled_contracts.json")


def load_dataset() -> dict:
    with open(DATASET, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate(use_llm: bool = False) -> dict:
    data = load_dataset()
    categories = data["categories"]

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
            print(f"LLM unavailable ({e}); running rule-only.")

    tp = fp = fn = 0
    per_cat = {c: {"tp": 0, "fn": 0} for c in categories}
    clean_docs = 0
    clean_false_positives = 0
    hard_neg_docs = 0
    hard_neg_false_positives = 0

    for item in data["contracts"]:
        report = review_contract(item["text"], contract_name=item["id"], generator=generator)
        predicted = {f.category for f in report.findings}
        expected = set(item["expected"])

        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

        for c in expected:
            if c in predicted:
                per_cat.setdefault(c, {"tp": 0, "fn": 0})["tp"] += 1
            else:
                per_cat.setdefault(c, {"tp": 0, "fn": 0})["fn"] += 1

        if not expected:
            clean_docs += 1
            if predicted:
                clean_false_positives += 1
            if "hard negative" in (item.get("note", "").lower()):
                hard_neg_docs += 1
                if predicted:
                    hard_neg_false_positives += 1

    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    clean_fp_rate = clean_false_positives / clean_docs if clean_docs else 0.0
    hard_neg_fp_rate = hard_neg_false_positives / hard_neg_docs if hard_neg_docs else 0.0

    return {
        "engine": "llm+rule" if generator else "rule",
        "n_contracts": len(data["contracts"]),
        "detection_rate_recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "clean_false_positive_rate": round(clean_fp_rate, 3),
        "hard_negative_false_positive_rate": round(hard_neg_fp_rate, 3),
        "n_hard_negatives": hard_neg_docs,
        "per_category_recall": {
            c: round(v["tp"] / (v["tp"] + v["fn"]), 3) if (v["tp"] + v["fn"]) else None
            for c, v in per_cat.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="Also run the LLM extraction pass (needs GOOGLE_API_KEY).")
    ap.add_argument("--min-recall", type=float, default=0.0, help="Exit non-zero if recall falls below this (for CI).")
    ap.add_argument("--min-precision", type=float, default=0.0, help="Exit non-zero if precision falls below this (for CI).")
    args = ap.parse_args()

    results = evaluate(use_llm=args.llm)

    print("\n=== Contract Risk Reviewer — Evaluation ===")
    print(f"Engine:                    {results['engine']}")
    print(f"Contracts:                 {results['n_contracts']}")
    print(f"Detection rate (recall):   {results['detection_rate_recall']:.1%}")
    print(f"Precision:                 {results['precision']:.1%}")
    print(f"F1:                        {results['f1']:.3f}")
    print(f"Clean-clause FP rate:      {results['clean_false_positive_rate']:.1%}")
    print(f"Hard-negative FP rate:     {results['hard_negative_false_positive_rate']:.1%}  (n={results['n_hard_negatives']})")
    print("\nPer-category recall:")
    for cat, r in results["per_category_recall"].items():
        if r is not None:
            print(f"  {cat:<26} {r:.0%}")
    print()

    failed = False
    if results["detection_rate_recall"] < args.min_recall:
        print(f"FAIL: recall {results['detection_rate_recall']:.3f} < min {args.min_recall}")
        failed = True
    if results["precision"] < args.min_precision:
        print(f"FAIL: precision {results['precision']:.3f} < min {args.min_precision}")
        failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
