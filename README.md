---
title: Contract Risk Reviewer
emoji: 📑
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.36.0"
python_version: "3.11"
app_file: streamlit_app.py
pinned: false
---

# 📑 Contract Risk Reviewer

An **agentic contract-review assistant**. Upload a contract and get a **risk report in seconds** — every risky or non-standard clause flagged with a **severity**, a plain-English rationale, a suggested redline, and the **exact cited text** it's based on.

The output is a *risk report, not a chatbot answer*. It's aimed at the people who actually review contracts by hand: small businesses and freelancers reviewing client paper, and lean legal/procurement teams.

## What it does

1. **Ingest** a PDF/TXT contract.
2. **Extract clauses** — split the contract into logical sections.
3. **Check risks** — a deterministic rule catalog *plus* an optional Gemini pass that catches non-standard clauses the rules miss.
4. **Report** — a typed `RiskReport`: overall risk rollup, executive summary, and per-clause findings with verbatim citations.

Runs **fully offline** with the rule engine (no API key needed). Add a Gemini key to enable the LLM pass.

## Agent pipeline

```
extract_clauses ──▶ rule_check ──▶ llm_check ──▶ build_report
```

Built with **LangGraph** ([`contract_agent.py`](contract_agent.py)); falls back to an equivalent linear run if LangGraph isn't installed. Every finding is validated against a **Pydantic schema** ([`contract_schema.py`](contract_schema.py)), so the report is always well-formed.

## Detected risk categories

Auto-renewal · Unlimited liability · Indemnification · Unilateral termination · IP assignment · Non-compete · Payment terms · Governing law / venue · Confidentiality · Warranty disclaimer

See [`risk_rules.py`](risk_rules.py) for the full catalog and patterns.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Upload a contract, or tick **"Use built-in sample contract"** for an instant demo. Optionally paste a Google Gemini API key in the sidebar to enable the LLM pass.

## Evaluation

The reviewer ships with a **labeled test set** ([`eval/labeled_contracts.json`](eval/labeled_contracts.json)) of risky and clean clauses, and a reproducible detection-rate harness:

```bash
python eval/run_eval.py          # rule engine only (offline, reproducible)
python eval/run_eval.py --llm    # also run the Gemini pass (needs GOOGLE_API_KEY)
```

Current rule-engine results on the labeled set:

| Metric | Value |
|---|---|
| Detection rate (recall) | **100%** |
| Precision | **100%** |
| F1 | **1.00** |
| Clean-clause false-positive rate | **0%** |

The eval reports **per-category recall** so patterns can be tuned as the labeled set grows. (The set is intentionally small and precision-oriented — it's the scaffolding for a detection metric, not a benchmark claim.)

## Tests

```bash
python -m pytest test_contract_reviewer.py -q
```

## Project layout

| File | Role |
|---|---|
| `contract_schema.py` | Pydantic models: `Severity`, `RiskFinding`, `Clause`, `RiskReport` |
| `risk_rules.py` | Deterministic risk catalog (offline baseline + LLM taxonomy) |
| `contract_agent.py` | LangGraph agent: extract → rule check → LLM check → report |
| `streamlit_app.py` | Upload → analyze → color-coded risk report UI |
| `eval/` | Labeled contracts + detection-rate harness |
| `document_processor.py`, `rag_engine.py`, `llm_interface.py` | Ingestion / hybrid retrieval / Gemini core |

## Ask-anything retrieval (bonus)

The bundled hybrid retrieval stack (FAISS + BM25 + TF-IDF + reranking) can also answer free-form questions about a contract (*"what's the notice period?"*) via the optional Gradio interface:

```bash
python run.py
```

## License

See [LICENSE](LICENSE).
