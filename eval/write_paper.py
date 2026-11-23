"""Render eval/research/PAPER.md from eval/research/summary_test.json + diagnostics (no hand-typed numbers).

Run after:  python eval/research_cuad.py --systems rules,llm_trunc,rt_llm,llm_trunc_ev,rt_llm_v2
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "eval" / "research"
S = json.loads((R / "summary_test.json").read_text())
sysd = S["systems"]
B = S["blind_spot"]
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731

# ---------------------------------------------------------------- diagnostics from the cache
from eval.research_cuad import load_gold, retrieve, SCORABLE_OURS, TRUNC  # noqa: E402
from contract_agent import extract_clauses  # noqa: E402

texts, labels, starts = load_gold("test")
cache = json.loads((R / "predictions_test.json").read_text())


def split_recall(system):
    out = {}
    for vis in (True, False):
        tp = n = 0
        for t, v in cache.items():
            for k in SCORABLE_OURS:
                if labels[t].get(k) and starts[t].get(k) and system in v:
                    if any(x < TRUNC for x in starts[t][k]) == vis:
                        n += 1; tp += v[system][k]
        out["inside" if vis else "beyond"] = (tp / n if n else 0.0, n)
    return out


hit = n = 0
for t in cache:
    cl = [x for x in extract_clauses(texts[t]) if len(x.text.strip()) > 40]
    cands, _ = retrieve(texts[t])
    for k in SCORABLE_OURS:
        if labels[t].get(k) and starts[t].get(k):
            spans = [(x.start_char, x.start_char + len(x.text)) for x in cl if x.text[:800] in cands.get(k, [])]
            n += 1; hit += any(a <= s < b for s in starts[t][k] for a, b in spans)
retrieval_hit = hit / n
vis_trunc, vis_rt2 = split_recall("llm_trunc_ev"), split_recall("rt_llm_v2")

from eval.research_cuad import _locate, SPANS  # noqa: E402
bd = {"grounded": 0, "elsewhere": 0, "unlocatable": 0, "empty": 0}
for t, v in cache.items():
    q = (v.get("llm_trunc_ev_meta") or {}).get("quotes", {})
    for k in SCORABLE_OURS:
        if v.get("llm_trunc_ev", {}).get(k) and labels[t].get(k):
            quote = q.get(k, "")
            if len(quote.strip()) < 15:
                bd["empty"] += 1; continue
            sp = _locate(texts[t], quote)
            if not sp:
                bd["unlocatable"] += 1
            elif any(a < sp[1] and sp[0] < b for a, b in SPANS[t][k]):
                bd["grounded"] += 1
            else:
                bd["elsewhere"] += 1
gold_total = sum(1 for t in cache for k in SCORABLE_OURS if labels[t].get(k))
upper = (bd["grounded"] + bd["unlocatable"]) / gold_total

NAMES = {"rules": "Rules (precision-first)",
         "llm_trunc": "LLM, single prompt, first 18K chars (shipped)",
         "llm_trunc_ev": "LLM, single prompt, first 18K chars (shipped, re-run with citations)",
         "rt_llm": "LLM, retrieval-targeted v1 (strict verifier, 800-char candidates)",
         "rt_llm_v2": "LLM, retrieval-targeted v2 (lenient verifier, full candidates)",
         "rules+llm_trunc": "Rules ∪ single-prompt LLM (shipped)",
         "rules+llm_trunc_ev": "Rules ∪ single-prompt LLM (re-run)",
         "rules+rt_llm": "Rules ∪ retrieval-targeted v1",
         "rules+rt_llm_v2": "Rules ∪ retrieval-targeted v2"}
order = [k for k in ["rules", "llm_trunc_ev", "rt_llm", "rt_llm_v2", "rules+llm_trunc_ev", "rules+rt_llm", "rules+rt_llm_v2"] if k in sysd]


def row(k):
    m = sysd[k]
    ci = m["ci95"]["f1"]
    c = m.get("cost", {})
    cost = f"{c.get('mean_in_tokens', 0):,.0f} / {c.get('median_latency_s', 0):.1f}s" if c else "— / <0.1s"
    return (f"| {NAMES[k]} | {m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{m['tp']} / {m['fp']} / {m['fn']} | {cost} |")


g_tr, g_rt = sysd.get("llm_trunc_ev@grounded", {}), sysd.get("rt_llm_v2@grounded", {})
best_f1 = max(order, key=lambda k: sysd[k]["f1"])
better_grounded = "retrieval-targeted v2" if g_rt.get("grounded_recall", 0) > g_tr.get("grounded_recall", 0) else "the single-prompt LLM"
mc = S.get("mcnemar", {})
tests = "\n".join(f"| {k} | {v['a_only_correct']} | {v['b_only_correct']} | {v['p_value']:.2g} |" for k, v in mc.items())
per_cat = "\n".join(
    f"| {c} | " + " | ".join(f"{sysd[k]['per_category'][c]['precision']:.2f} / {sysd[k]['per_category'][c]['recall']:.2f}"
                             for k in order) + " |" for c in sysd["rules"]["per_category"])

md = f"""# Contract-Level Scores Reward Guessing: Grounded Evaluation of LLM Contract Review on CUAD

**Karthik Mulugu** — draft, November 2026. Code: `eval/research_cuad.py`; local model `{S['model']}` ({S['backend']}, temperature 0, laptop GPU — contracts never leave the machine). Data: CUAD v1 test split ({S['n_contracts']} real commercial contracts, expert labels, CC BY 4.0).

## Abstract

LLM contract reviewers are commonly built as a single prompt over the contract, truncated to the model's context or budget, and evaluated by whether they report the right *kinds* of clauses per contract. On the CUAD test split, the single-prompt design reads the first {B['cutoff_chars']:,} characters, yet **{pct(B['share_beyond_cutoff'])} of expert-labeled risk clauses lie entirely beyond that cutoff**. It still "finds" {pct(vis_trunc['beyond'][0])} of those clauses it never saw, which is consistent with answering from priors about what contracts usually contain rather than from the text. On contract-level F1 the single-prompt LLM is the stronger design ({sysd['llm_trunc_ev']['f1']:.3f} vs. {sysd['rt_llm_v2']['f1']:.3f} for retrieval-targeted verification; McNemar p = {mc.get('rt_llm_v2 vs llm_trunc_ev', {}).get('p_value', float('nan')):.3f}). We therefore score **grounded recall**: a detection counts only if the passage the model cites overlaps the expert-labeled span. Under grounded scoring, the single-prompt LLM recovers {pct(g_tr.get('grounded_recall', 0))} of labeled clauses and retrieval-targeted verification — BM25 retrieves candidate clauses from the whole contract, the LLM verifies them — recovers {pct(g_rt.get('grounded_recall', 0))}. Retrieval itself places the labeled clause among the top-3 candidates {pct(retrieval_hit)} of the time, so the verifier, not retrieval, is the bottleneck; loosening the verifier (v1 → v2) moved contract-level F1 only from {sysd['rt_llm']['f1']:.3f} to {sysd['rt_llm_v2']['f1']:.3f} (not significant, p = {mc.get('rt_llm_v2 vs rt_llm', {}).get('p_value', float('nan')):.2f}), so a better verifier is the open problem. We recommend that contract-review systems be evaluated with grounded metrics, and we release the harness.

## 1. Problem

Small businesses and lean legal teams review counterparty paper by hand. A useful assistant must flag risky clauses (auto-renewal, uncapped liability, termination for convenience, IP assignment, non-compete, governing law) **and point to them**: a flag without the right citation cannot be checked, and a reviewer who trusts it is relying on a guess.

## 2. Setup

* **Labels.** For each contract and each of six CUAD-mapped categories, CUAD marks the clause present (answered question, with character spans) or absent. Contract-level metrics use presence; grounded metrics require the cited passage to overlap a labeled span.
* **Systems.** Rules (the project's deterministic engine); single-prompt LLM over the first {B['cutoff_chars']:,} characters, returning a quote per category; retrieval-targeted verification v1 (top-3 BM25 clauses per category, ≤800 characters each, strict "actually contains" verifier) and v2 (≤1,500 characters, lenient verifier that cites a passage id); and the union of rules with each LLM design, which is how the product combines them.
* **Statistics.** 95% bootstrap confidence intervals over contracts (2,000 resamples); exact McNemar tests on per-(contract, category) correctness.

## 3. Results

### 3.1 The truncation blind spot

| Contracts | Median length | Longer than cutoff | Labeled risk clauses beyond cutoff |
|---|---|---|---|
| {B['contracts']} | {B['median_chars']:,} chars | {pct(B['share_contracts_longer_than_cutoff'])} | **{pct(B['share_beyond_cutoff'])}** ({B['gold_clauses_entirely_beyond_cutoff']}/{B['gold_clauses']}) |

Contract-level recall split by whether the labeled clause is visible inside the cutoff:

| System | Clause inside cutoff | Clause entirely beyond cutoff |
|---|---|---|
| Single-prompt LLM | {pct(vis_trunc['inside'][0])} (n={vis_trunc['inside'][1]}) | {pct(vis_trunc['beyond'][0])} (n={vis_trunc['beyond'][1]}) |
| Retrieval-targeted v2 | {pct(vis_rt2['inside'][0])} | {pct(vis_rt2['beyond'][0])} |

A model that cannot see a clause and still reports it is answering from priors about what contracts usually contain.

### 3.2 Contract-level detection (6 categories × {S['n_contracts']} contracts)

| System | Precision | Recall | F1 [95% CI] | TP / FP / FN | Input tokens / median latency |
|---|---|---|---|---|---|
""" + "\n".join(row(k) for k in order) + f"""

### 3.3 Grounded recall (cited evidence overlaps the expert span)

| System | Grounded recall | Share of contract-level hits that are grounded |
|---|---|---|
| Single-prompt LLM | **{pct(g_tr.get('grounded_recall', 0))}** | {pct(g_tr.get('share_of_true_positives_grounded', 0))} ({g_tr.get('grounded_tp', 0)}/{g_tr.get('tp', 0)}) |
| Retrieval-targeted v2 | **{pct(g_rt.get('grounded_recall', 0))}** | {pct(g_rt.get('share_of_true_positives_grounded', 0))} ({g_rt.get('grounded_tp', 0)}/{g_rt.get('tp', 0)}) |

Why the single-prompt hits are ungrounded: of its {bd['grounded'] + bd['elsewhere'] + bd['unlocatable'] + bd['empty']} correct contract-level detections, {bd['grounded']} cite the labeled passage, **{bd['elsewhere']} cite a real passage that is not the labeled clause**, {bd['unlocatable']} cite text that cannot be found verbatim in the contract (paraphrased or invented) and {bd['empty']} cite nothing. Even if every unlocatable quote were counted as correct, its grounded recall would be at most {pct(upper)}, still below retrieval-targeted v2.

Retrieval places the labeled clause among the top-3 candidates for **{pct(retrieval_hit)}** of labeled clauses, so most of the remaining gap is the verifier's.

### 3.4 Per category (precision / recall)

| Category | """ + " | ".join(NAMES[k] for k in order) + """ |
|---|""" + "---|" * len(order) + f"""
{per_cat}

### 3.5 Significance (exact McNemar, per contract × category)

| Comparison | Only A correct | Only B correct | p |
|---|---|---|---|
{tests}

## 4. Discussion

On contract-level F1 the best system is **{NAMES[best_f1]} ({sysd[best_f1]['f1']:.3f})**. On grounded recall — the share of labeled clauses a reviewer could actually verify from the citation — the stronger design is **{better_grounded}**. The two views disagree because contract-level presence can be earned without reading the clause. For a product whose value is pointing a busy reviewer at the right paragraph, the grounded number is the one that matters, and it is the one current evaluations of LLM contract review rarely report.

## 5. Limitations

One local 8B model at temperature 0; contract-level labels for six of our ten categories; CUAD's Governing Law category labels any governing-law clause while our rules flag only exclusive venue (lower rule recall by design); BM25 queries written once from category definitions and not tuned on the test split; grounding by span overlap accepts a citation that overlaps any labeled span for the category, and model quotes are located by exact (whitespace-normalized) match, which undercounts paraphrased citations.

## 6. Reproduce

```bash
python eval/research_cuad.py --systems rules,llm_trunc,rt_llm,llm_trunc_ev,rt_llm_v2
python eval/write_paper.py
```
"""
(R / "PAPER.md").write_text(md, encoding="utf-8")
print("wrote", R / "PAPER.md")
