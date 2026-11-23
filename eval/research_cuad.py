"""Retrieval-targeted LLM verification for contract risk review — CUAD study.

Research questions
  RQ1  How much of a real contract does a single-prompt LLM reviewer actually read?
       (We measure how many expert-labeled CUAD clauses sit beyond the prompt cutoff.)
  RQ2  Does retrieval-targeted verification — retrieve the few clauses most likely to
       contain each risk from the WHOLE contract, then ask the LLM to verify only those —
       beat (a) precision-first rules, (b) a truncated full-prompt LLM, and (c) their union,
       on contract-level precision, recall and F1?
  RQ3  What does each design cost per contract (tokens, latency)?

Ground truth: CUAD v1 test split (102 real contracts, expert annotations, CC BY 4.0). For each
contract and each of our 6 CUAD-mappable categories, CUAD says whether the clause is present
(answered question) or absent (is_impossible) -> a full 2x2 per (contract, category), so we can
report precision as well as recall.

Usage
  python eval/research_cuad.py --systems rules                 # free, all 102 test contracts
  python eval/research_cuad.py --systems rules,llm_trunc,rt_llm --limit 10   # smoke test
  python eval/research_cuad.py                                  # full study (Gemini calls)
Outputs eval/research/ (per-contract predictions, summary JSON, figures).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract_agent import review_contract, extract_clauses  # noqa: E402
from eval.cuad_mapping import CUAD_TO_OURS, SCORABLE_OURS  # noqa: E402
from eval.run_cuad_eval import _load_cuad, _category_from_question  # noqa: E402

OUT = ROOT / "eval" / "research"
BACKEND = os.environ.get("LLM_BACKEND", "ollama")          # "ollama" (local, default) or "gemini"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b") if BACKEND == "ollama" else os.environ.get("RESEARCH_MODEL", "gemini-2.5-flash")
TRUNC = 18_000          # what the shipped single-prompt LLM pass sees (contract_agent.llm_extract)
TOP_K = 3               # candidate clauses retrieved per category
CAND_CHARS = 800        # characters kept per candidate (keeps the prompt shorter than the truncated one)
CAT_QUERIES = {         # retrieval queries: how each risk is actually worded in contracts
    "Auto-renewal": "automatically renew renewal term successive periods unless notice of non-renewal",
    "Unlimited liability": "liability shall not be limited unlimited uncapped damages exclusion limitation of liability not apply",
    "Unilateral termination": "terminate for convenience at any time without cause upon written notice",
    "IP assignment": "assigns all right title and interest intellectual property work product inventions",
    "Non-compete": "shall not compete engage in competing business non-competition restrictive covenant",
    "Governing law / venue": "governed by the laws of the state exclusive jurisdiction venue courts",
}
CAT_DEFS = {
    "Auto-renewal": "the contract renews automatically for further terms unless a party gives notice",
    "Unlimited liability": "a party's liability is uncapped / not limited (or a cap explicitly does not apply to some liability)",
    "Unilateral termination": "a party may terminate for convenience / without cause",
    "IP assignment": "intellectual property or work product is assigned or transferred to the counterparty",
    "Non-compete": "a party is restricted from competing or operating in a market / line of business",
    "Governing law / venue": "the contract names a governing law (any jurisdiction)",
}


# --------------------------------------------------------------------------- data
SPANS = defaultdict(lambda: defaultdict(list))   # title -> category -> [(start, end)] expert answer spans


def load_gold(split: str):
    """Return {title: text}, {title: {cat: present?}}, {title: {cat: [answer_start,...]}}."""
    data = _load_cuad(split)
    texts, labels, starts = {}, defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for entry in data:
        t = entry["title"]
        for para in entry["paragraphs"]:
            texts.setdefault(t, para["context"])
            for qa in para["qas"]:
                cuad_cat = _category_from_question(qa["question"])
                if cuad_cat not in CUAD_TO_OURS:
                    continue
                ours = CUAD_TO_OURS[cuad_cat]
                present = (not qa.get("is_impossible")) and bool(qa.get("answers"))
                labels[t][ours] = labels[t].get(ours, False) or present
                for a in qa.get("answers", []):
                    starts[t][ours].append(int(a["answer_start"]))
                    SPANS[t][ours].append((int(a["answer_start"]), int(a["answer_start"]) + len(a.get("text", ""))))
    return texts, labels, starts


# --------------------------------------------------------------------------- systems
_GEN = None


def gemini():
    global _GEN
    if _GEN is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _GEN = genai.GenerativeModel(MODEL, generation_config={"temperature": 0.0,
                                                               "response_mime_type": "application/json"})
    return _GEN


def _ollama_json(prompt: str, num_ctx: int = 12288) -> tuple[dict | None, dict]:
    """Local backend: qwen3:8b on Ollama, temperature 0, JSON mode, thinking off."""
    import urllib.request
    t0 = time.time()
    body = {"model": os.environ.get("OLLAMA_MODEL", "qwen3:8b"), "think": False, "stream": False,
            "format": "json", "options": {"temperature": 0, "num_ctx": num_ctx, "seed": 0},
            "messages": [{"role": "user", "content": prompt}]}
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=900).read())
        raw = d["message"]["content"]
        meta = {"latency_s": round(time.time() - t0, 2), "in_tokens": d.get("prompt_eval_count", 0),
                "out_tokens": d.get("eval_count", 0)}
        s, e = raw.find("{"), raw.rfind("}")
        return (json.loads(raw[s:e + 1]) if s != -1 else None), meta
    except Exception as ex:  # noqa: BLE001
        return None, {"latency_s": round(time.time() - t0, 2), "error": str(ex)[:200]}


_LAST = [0.0]
MIN_GAP_S = float(os.environ.get("RESEARCH_MIN_GAP_S", "4.5"))   # stay under free-tier RPM


def call_llm(prompt: str) -> tuple[dict | None, dict]:
    if BACKEND == "ollama":
        return _ollama_json(prompt)
    wait = MIN_GAP_S - (time.time() - _LAST[0])
    if wait > 0:
        time.sleep(wait)
    _LAST[0] = time.time()
    t0 = time.time()
    for attempt in range(5):
        try:
            r = gemini().generate_content(prompt)
            u = getattr(r, "usage_metadata", None)
            meta = {"latency_s": round(time.time() - t0, 2),
                    "in_tokens": getattr(u, "prompt_token_count", 0) if u else 0,
                    "out_tokens": getattr(u, "candidates_token_count", 0) if u else 0}
            raw = (r.text or "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            return (json.loads(raw[s:e + 1]) if s != -1 else None), meta
        except Exception as ex:  # noqa: BLE001
            if "429" in str(ex) or "quota" in str(ex).lower() or "503" in str(ex):
                time.sleep(8 * (attempt + 1)); continue
            return None, {"latency_s": round(time.time() - t0, 2), "error": str(ex)[:200]}
    return None, {"latency_s": round(time.time() - t0, 2), "error": "retries exhausted"}


def sys_rules(text: str):
    found = {f.category for f in review_contract(text, generator=None).findings}
    return {c: c in found for c in SCORABLE_OURS}, {}


def sys_llm_trunc(text: str):
    """The shipped design: one prompt with the first TRUNC characters."""
    cats = "\n".join(f"- {c}: {d}" for c, d in CAT_DEFS.items())
    prompt = (f"You review commercial contracts. For each category below, decide if the contract "
              f"contains such a clause.\n{cats}\n\nReturn JSON: {{\"<category>\": {{\"present\": true|false, "
              f"\"quote\": \"exact text or empty\"}}, ...}} using exactly these category names.\n\n"
              f"---- CONTRACT ----\n{text[:TRUNC]}\n---- END ----")
    data, meta = call_llm(prompt)
    meta["quotes"] = {c: str(((data or {}).get(c) or {}).get("quote") or "")[:600] for c in SCORABLE_OURS}
    return {c: bool(((data or {}).get(c) or {}).get("present")) for c in SCORABLE_OURS}, meta


def retrieve(text: str, k: int = TOP_K):
    from rank_bm25 import BM25Okapi
    clauses = [c.text for c in extract_clauses(text) if len(c.text.strip()) > 40]
    if not clauses:
        clauses = [text[i:i + 1500] for i in range(0, len(text), 1500)]
    tok = lambda s: re.findall(r"[a-z]+", s.lower())  # noqa: E731
    bm = BM25Okapi([tok(c) for c in clauses])
    out = {}
    for cat, q in CAT_QUERIES.items():
        scores = bm.get_scores(tok(q))
        idx = np.argsort(scores)[::-1][:k]
        out[cat] = [clauses[i][:CAND_CHARS] for i in idx if scores[i] > 0]
    return out, len(clauses)


def sys_rt_llm(text: str):
    """Proposed: retrieval-targeted verification over the whole contract (one short call)."""
    cands, n_clauses = retrieve(text)
    blocks = []
    for cat, cl in cands.items():
        body = "\n".join(f"  [{cat[:3].upper()}-{i}] {c}" for i, c in enumerate(cl)) or "  (no candidates)"
        blocks.append(f"### {cat} — {CAT_DEFS[cat]}\n{body}")
    prompt = ("You verify risk clauses in a commercial contract. For each category you get the few "
              "passages most likely to contain it, retrieved from the full contract. Decide whether "
              "ANY passage actually contains that kind of clause (not merely mentions the topic). "
              "Return JSON: {\"<category>\": {\"present\": true|false, \"evidence_id\": \"<id or empty>\"}, ...} "
              "using exactly the category names.\n\n" + "\n\n".join(blocks))
    data, meta = call_llm(prompt)
    meta["n_clauses"] = n_clauses
    return {c: bool(((data or {}).get(c) or {}).get("present")) for c in SCORABLE_OURS}, meta


def _locate(text: str, quote: str):
    """Character span of a model-cited quote in the contract (whitespace-insensitive), or None."""
    q = re.sub(r"\s+", " ", (quote or "")).strip()
    if len(q) < 15:
        return None
    flat, idx = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            if flat and flat[-1] == " ":
                continue
            flat.append(" ")
        else:
            flat.append(ch)
        idx.append(i)
    flat = "".join(flat)
    for probe in (q, q[:120], q[:60]):
        j = flat.find(probe)
        if j != -1:
            return idx[j], idx[min(j + len(q), len(idx) - 1)]
    return None


def sys_llm_trunc_ev(text: str):
    """Shipped design, re-run so the cited quote is kept (for grounded scoring)."""
    return sys_llm_trunc(text)


def retrieve_spans(text: str, k: int = 3, chars: int = 1500):
    from rank_bm25 import BM25Okapi
    cl = [c for c in extract_clauses(text) if len(c.text.strip()) > 40]
    if not cl:
        return {}, 0
    tok = lambda s: re.findall(r"[a-z]+", s.lower())  # noqa: E731
    bm = BM25Okapi([tok(c.text) for c in cl])
    out = {}
    for cat, q in CAT_QUERIES.items():
        sc = bm.get_scores(tok(q))
        out[cat] = [(cl[i].text[:chars], cl[i].start_char) for i in np.argsort(sc)[::-1][:k] if sc[i] > 0]
    return out, len(cl)


def sys_rt_llm_v2(text: str):
    """Retrieval-targeted verification v2: full-length candidates (<=1500 chars), lenient verifier,
    and the cited passage is returned so detections can be grounded."""
    cands, n_clauses = retrieve_spans(text)
    blocks, ids = [], {}
    for cat, cl in cands.items():
        lines = []
        for i, (c, st) in enumerate(cl):
            cid = f"{cat[:3].upper()}-{i}"
            ids[cid] = (c, st)
            lines.append(f"  [{cid}] {c}")
        blocks.append(f"### {cat} - {CAT_DEFS[cat]}\n" + ("\n".join(lines) or "  (no candidates)"))
    prompt = ("You review a commercial contract. For each category you get the passages most likely to "
              "contain it, retrieved from anywhere in the contract. Answer present=true if any passage "
              "establishes that kind of clause, even if it is worded unusually or only part of the passage "
              "is relevant. Cite the passage id. Return JSON: {\"<category>\": {\"present\": true|false, "
              "\"evidence_id\": \"<id or empty>\"}, ...} using exactly the category names.\n\n" + "\n\n".join(blocks))
    data, meta = call_llm(prompt)
    meta["n_clauses"] = n_clauses
    pred, ev = {}, {}
    for c in SCORABLE_OURS:
        d = (data or {}).get(c) or {}
        pred[c] = bool(d.get("present"))
        eid = str(d.get("evidence_id") or "").strip("[] ")
        if eid in ids:
            ev[c] = [ids[eid][1], ids[eid][1] + len(ids[eid][0])]
    meta["evidence_spans"] = ev
    return pred, meta


def union(a: dict, b: dict) -> dict:
    return {c: a[c] or b[c] for c in SCORABLE_OURS}


# --------------------------------------------------------------------------- metrics
def prf(pairs):
    tp = sum(p and g for p, g in pairs); fp = sum(p and not g for p, g in pairs)
    fn = sum((not p) and g for p, g in pairs)
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def bootstrap(per_contract, system, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    titles = list(per_contract)
    vals = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        samp = rng.choice(titles, size=len(titles), replace=True)
        m = prf([(per_contract[t][system][c], per_contract[t]["gold"][c]) for t in samp for c in SCORABLE_OURS])
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def mcnemar(per_contract, a, b):
    """Exact McNemar on per-(contract,category) correctness."""
    from scipy.stats import binomtest
    n01 = n10 = 0
    for t in per_contract:
        for c in SCORABLE_OURS:
            g = per_contract[t]["gold"][c]
            ca, cb = per_contract[t][a][c] == g, per_contract[t][b][c] == g
            n01 += (ca and not cb); n10 += (cb and not ca)
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "full"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--systems", default="rules,llm_trunc,rt_llm")
    a = ap.parse_args()
    systems = a.systems.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    texts, labels, starts = load_gold(a.split)
    titles = list(texts)[: a.limit] if a.limit else list(texts)

    # RQ1: truncation blind spot
    gold_n = beyond = 0; lens = []
    for t in titles:
        lens.append(len(texts[t]))
        for c, ss in starts[t].items():
            if labels[t].get(c):
                gold_n += 1
                beyond += all(s >= TRUNC for s in ss)
    blind = {"contracts": len(titles), "median_chars": int(np.median(lens)),
             "share_contracts_longer_than_cutoff": float(np.mean(np.array(lens) > TRUNC)),
             "gold_clauses": gold_n, "gold_clauses_entirely_beyond_cutoff": beyond,
             "share_beyond_cutoff": beyond / gold_n if gold_n else 0.0, "cutoff_chars": TRUNC}
    print("RQ1 truncation blind spot:", json.dumps(blind))

    cache_p = OUT / f"predictions_{a.split}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    fns = {"rules": sys_rules, "llm_trunc": sys_llm_trunc, "rt_llm": sys_rt_llm,
           "llm_trunc_ev": sys_llm_trunc_ev, "rt_llm_v2": sys_rt_llm_v2}
    per, costs = {}, defaultdict(list)
    for i, t in enumerate(titles, 1):
        rec = cache.get(t, {})
        for s in systems:
            if s not in rec or (rec.get(f"{s}_meta") or {}).get("error"):
                pred, meta = fns[s](texts[t])
                rec[s] = pred; rec[f"{s}_meta"] = meta
        cache[t] = rec
        if i % 5 == 0:
            cache_p.write_text(json.dumps(cache))
            print(f"  {i}/{len(titles)}", flush=True)
        row = {"gold": {c: bool(labels[t].get(c, False)) for c in SCORABLE_OURS}}
        for s in systems:
            row[s] = rec[s]
            if rec.get(f"{s}_meta"):
                costs[s].append(rec[f"{s}_meta"])
        if "rules" in systems and "llm_trunc" in systems:
            row["rules+llm_trunc"] = union(rec["rules"], rec["llm_trunc"])
        if "rules" in systems and "rt_llm" in systems:
            row["rules+rt_llm"] = union(rec["rules"], rec["rt_llm"])
        if "rules" in systems and "rt_llm_v2" in systems:
            row["rules+rt_llm_v2"] = union(rec["rules"], rec["rt_llm_v2"])
        if "rules" in systems and "llm_trunc_ev" in systems:
            row["rules+llm_trunc_ev"] = union(rec["rules"], rec["llm_trunc_ev"])
        # grounded detections: the model's cited evidence overlaps an expert answer span
        for s_ in ("llm_trunc_ev", "rt_llm_v2"):
            if s_ in systems:
                meta_ = rec.get(f"{s_}_meta") or {}
                g = {}
                for c in SCORABLE_OURS:
                    if s_ == "rt_llm_v2":
                        sp = (meta_.get("evidence_spans") or {}).get(c)
                    else:
                        sp = _locate(texts[t], (meta_.get("quotes") or {}).get(c, ""))
                    g[c] = bool(rec[s_][c] and sp and any(a0 < sp[1] and sp[0] < b0 for a0, b0 in SPANS[t][c]))
                row[s_ + "@grounded"] = g
        per[t] = row
    cache_p.write_text(json.dumps(cache))

    sys_names = [k for k in next(iter(per.values())) if k != "gold"]
    summary = {"backend": BACKEND, "model": MODEL, "split": a.split, "n_contracts": len(titles), "blind_spot": blind, "systems": {}}
    for s in [k for k in sys_names if not k.endswith("@grounded")]:
        m = prf([(per[t][s][c], per[t]["gold"][c]) for t in per for c in SCORABLE_OURS])
        m["ci95"] = bootstrap(per, s)
        m["per_category"] = {c: prf([(per[t][s][c], per[t]["gold"][c]) for t in per]) for c in SCORABLE_OURS}
        base = s.split("+")[-1]
        if costs.get(base):
            m["cost"] = {"mean_in_tokens": float(np.mean([x.get("in_tokens", 0) for x in costs[base]])),
                         "mean_out_tokens": float(np.mean([x.get("out_tokens", 0) for x in costs[base]])),
                         "median_latency_s": float(np.median([x.get("latency_s", 0) for x in costs[base]])),
                         "errors": sum(1 for x in costs[base] if x.get("error"))}
        summary["systems"][s] = m
        print(f"{s:16s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  CI(F1)={m['ci95']['f1']}")
    for s in [k for k in sys_names if k.endswith("@grounded")]:
        base = s.split("@")[0]
        gold_n = sum(per[t]["gold"][c] for t in per for c in SCORABLE_OURS)
        gtp = sum(per[t][s][c] and per[t]["gold"][c] for t in per for c in SCORABLE_OURS)
        tp = sum(per[t][base][c] and per[t]["gold"][c] for t in per for c in SCORABLE_OURS)
        summary["systems"][s] = {"grounded_recall": gtp / gold_n if gold_n else 0.0,
                                 "share_of_true_positives_grounded": gtp / tp if tp else 0.0,
                                 "grounded_tp": gtp, "tp": tp}
        print(f"{s:22s} grounded recall={gtp / gold_n:.3f}  grounded share of TPs={gtp / max(1, tp):.3f}")
    tests = {}
    for a_, b_ in [("rules+rt_llm", "rules+llm_trunc"), ("rules+rt_llm", "rules"), ("rt_llm", "llm_trunc"),
                   ("rt_llm_v2", "llm_trunc_ev"), ("rules+rt_llm_v2", "rules+llm_trunc_ev"), ("rt_llm_v2", "rt_llm")]:
        if a_ in sys_names and b_ in sys_names:
            tests[f"{a_} vs {b_}"] = mcnemar(per, a_, b_)
    summary["mcnemar"] = tests
    print("McNemar:", json.dumps(tests))
    (OUT / f"summary_{a.split}.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT / f"summary_{a.split}.json")


if __name__ == "__main__":
    main()
