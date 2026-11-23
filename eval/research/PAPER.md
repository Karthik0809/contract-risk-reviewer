# Contract-Level Scores Reward Guessing: Grounded Evaluation of LLM Contract Review on CUAD

**Karthik Mulugu** — draft, November 2026. Code: `eval/research_cuad.py`; local model `qwen3:8b` (ollama, temperature 0, laptop GPU — contracts never leave the machine). Data: CUAD v1 test split (102 real commercial contracts, expert labels, CC BY 4.0).

## Abstract

LLM contract reviewers are commonly built as a single prompt over the contract, truncated to the model's context or budget, and evaluated by whether they report the right *kinds* of clauses per contract. On the CUAD test split, the single-prompt design reads the first 18,000 characters, yet **51.8% of expert-labeled risk clauses lie entirely beyond that cutoff**. It still "finds" 48.0% of those clauses it never saw, which is consistent with answering from priors about what contracts usually contain rather than from the text. On contract-level F1 the single-prompt LLM is the stronger design (0.702 vs. 0.623 for retrieval-targeted verification; McNemar p = 0.004). We therefore score **grounded recall**: a detection counts only if the passage the model cites overlaps the expert-labeled span. Under grounded scoring, the single-prompt LLM recovers 34.5% of labeled clauses and retrieval-targeted verification — BM25 retrieves candidate clauses from the whole contract, the LLM verifies them — recovers 46.7%. Retrieval itself places the labeled clause among the top-3 candidates 87.8% of the time, so the verifier, not retrieval, is the bottleneck; loosening the verifier (v1 → v2) moved contract-level F1 only from 0.607 to 0.623 (not significant, p = 0.80), so a better verifier is the open problem. We recommend that contract-review systems be evaluated with grounded metrics, and we release the harness.

## 1. Problem

Small businesses and lean legal teams review counterparty paper by hand. A useful assistant must flag risky clauses (auto-renewal, uncapped liability, termination for convenience, IP assignment, non-compete, governing law) **and point to them**: a flag without the right citation cannot be checked, and a reviewer who trusts it is relying on a guess.

## 2. Setup

* **Labels.** For each contract and each of six CUAD-mapped categories, CUAD marks the clause present (answered question, with character spans) or absent. Contract-level metrics use presence; grounded metrics require the cited passage to overlap a labeled span.
* **Systems.** Rules (the project's deterministic engine); single-prompt LLM over the first 18,000 characters, returning a quote per category; retrieval-targeted verification v1 (top-3 BM25 clauses per category, ≤800 characters each, strict "actually contains" verifier) and v2 (≤1,500 characters, lenient verifier that cites a passage id); and the union of rules with each LLM design, which is how the product combines them.
* **Statistics.** 95% bootstrap confidence intervals over contracts (2,000 resamples); exact McNemar tests on per-(contract, category) correctness.

## 3. Results

### 3.1 The truncation blind spot

| Contracts | Median length | Longer than cutoff | Labeled risk clauses beyond cutoff |
|---|---|---|---|
| 102 | 25,657 chars | 67.6% | **51.8%** (102/197) |

Contract-level recall split by whether the labeled clause is visible inside the cutoff:

| System | Clause inside cutoff | Clause entirely beyond cutoff |
|---|---|---|
| Single-prompt LLM | 82.1% (n=95) | 48.0% (n=102) |
| Retrieval-targeted v2 | 69.5% | 52.0% |

A model that cannot see a clause and still reports it is answering from priors about what contracts usually contain.

### 3.2 Contract-level detection (6 categories × 102 contracts)

| System | Precision | Recall | F1 [95% CI] | TP / FP / FN | Input tokens / median latency |
|---|---|---|---|---|---|
| Rules (precision-first) | 0.798 | 0.381 | **0.515** [0.441, 0.581] | 75 / 19 / 122 | — / <0.1s |
| LLM, single prompt, first 18K chars (shipped, re-run with citations) | 0.770 | 0.645 | **0.702** [0.651, 0.751] | 127 / 38 / 70 | 3,408 / 15.8s |
| LLM, retrieval-targeted v1 (strict verifier, 800-char candidates) | 0.673 | 0.553 | **0.607** [0.538, 0.665] | 109 / 53 / 88 | 2,811 / 8.4s |
| LLM, retrieval-targeted v2 (lenient verifier, full candidates) | 0.643 | 0.604 | **0.623** [0.565, 0.680] | 119 / 66 / 78 | 4,462 / 16.8s |
| Rules ∪ single-prompt LLM (re-run) | 0.733 | 0.751 | **0.742** [0.695, 0.787] | 148 / 54 / 49 | 3,408 / 15.8s |
| Rules ∪ retrieval-targeted v1 | 0.668 | 0.695 | **0.682** [0.627, 0.733] | 137 / 68 / 60 | 2,811 / 8.4s |
| Rules ∪ retrieval-targeted v2 | 0.647 | 0.736 | **0.689** [0.636, 0.739] | 145 / 79 / 52 | 4,462 / 16.8s |

### 3.3 Grounded recall (cited evidence overlaps the expert span)

| System | Grounded recall | Share of contract-level hits that are grounded |
|---|---|---|
| Single-prompt LLM | **34.5%** | 53.5% (68/127) |
| Retrieval-targeted v2 | **46.7%** | 77.3% (92/119) |

Why the single-prompt hits are ungrounded: of its 127 correct contract-level detections, 68 cite the labeled passage, **40 cite a real passage that is not the labeled clause**, 15 cite text that cannot be found verbatim in the contract (paraphrased or invented) and 4 cite nothing. Even if every unlocatable quote were counted as correct, its grounded recall would be at most 42.1%, still below retrieval-targeted v2.

Retrieval places the labeled clause among the top-3 candidates for **87.8%** of labeled clauses, so most of the remaining gap is the verifier's.

### 3.4 Per category (precision / recall)

| Category | Rules (precision-first) | LLM, single prompt, first 18K chars (shipped, re-run with citations) | LLM, retrieval-targeted v1 (strict verifier, 800-char candidates) | LLM, retrieval-targeted v2 (lenient verifier, full candidates) | Rules ∪ single-prompt LLM (re-run) | Rules ∪ retrieval-targeted v1 | Rules ∪ retrieval-targeted v2 |
|---|---|---|---|---|---|---|---|
| Auto-renewal | 1.00 / 0.42 | 0.94 / 0.62 | 0.44 / 0.65 | 0.41 / 0.58 | 0.95 / 0.73 | 0.46 / 0.73 | 0.46 / 0.73 |
| Governing law / venue | 1.00 / 0.24 | 0.93 / 0.80 | 0.96 / 0.58 | 0.95 / 0.66 | 0.93 / 0.82 | 0.97 / 0.67 | 0.95 / 0.75 |
| IP assignment | 0.81 / 0.57 | 0.66 / 0.83 | 0.59 / 0.57 | 0.58 / 0.61 | 0.62 / 0.87 | 0.60 / 0.78 | 0.61 / 0.83 |
| Non-compete | 0.64 / 0.78 | 0.65 / 0.48 | 0.83 / 0.43 | 0.81 / 0.57 | 0.60 / 0.91 | 0.67 / 0.87 | 0.68 / 0.91 |
| Unilateral termination | 0.73 / 0.38 | 0.50 / 0.52 | 0.62 / 0.69 | 0.54 / 0.72 | 0.49 / 0.62 | 0.61 / 0.76 | 0.52 / 0.76 |
| Unlimited liability | 0.50 / 0.15 | 0.00 / 0.00 | 0.14 / 0.08 | 0.09 / 0.08 | 0.40 / 0.15 | 0.20 / 0.15 | 0.14 / 0.15 |

### 3.5 Significance (exact McNemar, per contract × category)

| Comparison | Only A correct | Only B correct | p |
|---|---|---|---|
| rules+rt_llm vs rules+llm_trunc | 45 | 71 | 0.02 |
| rules+rt_llm vs rules | 62 | 49 | 0.25 |
| rt_llm vs llm_trunc | 56 | 90 | 0.0061 |
| rt_llm_v2 vs llm_trunc_ev | 57 | 93 | 0.0041 |
| rules+rt_llm_v2 vs rules+llm_trunc_ev | 46 | 74 | 0.013 |
| rt_llm_v2 vs rt_llm | 30 | 33 | 0.8 |

## 4. Discussion

On contract-level F1 the best system is **Rules ∪ single-prompt LLM (re-run) (0.742)**. On grounded recall — the share of labeled clauses a reviewer could actually verify from the citation — the stronger design is **retrieval-targeted v2**. The two views disagree because contract-level presence can be earned without reading the clause. For a product whose value is pointing a busy reviewer at the right paragraph, the grounded number is the one that matters, and it is the one current evaluations of LLM contract review rarely report.

## 5. Limitations

One local 8B model at temperature 0; contract-level labels for six of our ten categories; CUAD's Governing Law category labels any governing-law clause while our rules flag only exclusive venue (lower rule recall by design); BM25 queries written once from category definitions and not tuned on the test split; grounding by span overlap accepts a citation that overlaps any labeled span for the category, and model quotes are located by exact (whitespace-normalized) match, which undercounts paraphrased citations.

## 6. Reproduce

```bash
python eval/research_cuad.py --systems rules,llm_trunc,rt_llm,llm_trunc_ev,rt_llm_v2
python eval/write_paper.py
```
