# RAG Engine - FAISS + BM25 + TF-IDF; optional Pinecone when API key is set
import re
import hashlib
import time
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TARGET_TOKENS, OVERLAP_SENTENCES, BOUNDARY_DROP, TOK_CHARS_RATIO
from config import PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION
from config import USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM
from utils import safe_sent_tokenize, DomainHeuristics

# Optional Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False


class OpenAIEmbedder:
    """Wrapper so OpenAI text-embedding-3-large can be used like SentenceTransformer (e.g. for 1024-dim Pinecone index)."""
    def __init__(self, api_key=None, model=OPENAI_EMBEDDING_MODEL, dimension=OPENAI_EMBEDDING_DIM):
        self._dim = dimension
        self._model = model
        self._client = None
        if api_key and (api_key or "").strip():
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=(api_key or "").strip())
            except Exception:
                pass

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32, **kwargs):
        if not self._client:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
        if isinstance(texts, str):
            texts = [texts]
        import numpy as np
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = self._client.embeddings.create(model=self._model, input=batch, dimensions=self._dim)
            for e in sorted(r.data, key=lambda x: x.index):
                out.append(e.embedding)
        arr = np.array(out, dtype="float32")
        if normalize_embeddings and len(arr) > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            arr = arr / norms
        return arr


class HashingEmbedder:
    """Deterministic local embedder that requires no external downloads."""
    def __init__(self, dim=384):
        self._dim = int(dim)

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for txt in texts:
            vec = np.zeros(self._dim, dtype="float32")
            tokens = re.findall(r"[a-z0-9]+", (txt or "").lower())
            if not tokens:
                out.append(vec)
                continue
            for tok in tokens:
                h = hashlib.md5(tok.encode("utf-8")).hexdigest()
                idx = int(h[:8], 16) % self._dim
                sign = -1.0 if (int(h[8:10], 16) % 2) else 1.0
                vec[idx] += sign
            if normalize_embeddings:
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
            out.append(vec)
        return np.array(out, dtype="float32")


class HierarchicalChunker:
    def __init__(self, embed_model, target_tokens=TARGET_TOKENS, overlap_sentences=OVERLAP_SENTENCES, boundary_drop=BOUNDARY_DROP, tok_chars_ratio=TOK_CHARS_RATIO):
        self.emb = embed_model
        self.target_tokens = target_tokens
        self.overlap = overlap_sentences
        self.boundary_drop = boundary_drop
        self.tok_chars_ratio = tok_chars_ratio

    def _est_tokens(self, s):
        return max(1, int(len(s) * self.tok_chars_ratio))

    def split(self, raw_text, meta):
        text = re.sub(r'--- Page \d+.*?---\n?', '', raw_text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.(\s+)([A-Z])', r'.\n\2', text)
        chunks = []
        # Faster default chunking profile: fewer scales to reduce processing time.
        for size, label in [(300, "medium"), (150, "small")]:
            chunks.extend(self._mk_chunks(text, meta, size, label))
        return chunks

    def _mk_chunks(self, text, meta, target_tokens, chunk_type):
        sents = safe_sent_tokenize(text) or [text]
        if len(sents) < 2:
            return [{"text": text, "metadata": {"chunk_id": 0, "chunk_type": chunk_type,
                                                "source_file": meta.get("file"), "source_path": meta.get("path"),
                                                "chunk_hash": hashlib.md5(text.encode()).hexdigest()[:10]}}]
        sent_vecs = self.emb.encode(sents, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        chunks = []
        cur, curv, prev = [], [], None

        def flush():
            if not cur:
                return
            t = " ".join(cur).strip()
            chunks.append({"text": t, "metadata": {
                "chunk_id": len(chunks), "chunk_type": chunk_type,
                "source_file": meta.get("file"), "source_path": meta.get("path"),
                "chunk_hash": hashlib.md5(t.encode()).hexdigest()[:10],
                "token_count": self._est_tokens(t), "sentence_count": len(cur)
            }})

        for s, v in zip(sents, sent_vecs):
            if not cur:
                cur, curv, prev = [s], [v], 1.0
                continue
            sim = float(np.dot(v, curv[-1]))
            budget = self._est_tokens(" ".join(cur + [s]))
            boundary = (prev - sim) > self.boundary_drop or budget > target_tokens
            if boundary:
                flush()
                ov = cur[-self.overlap:] if len(cur) >= self.overlap else cur
                cur = ov + [s]
                curv = ([self.emb.encode([o], normalize_embeddings=True)[0] for o in ov] if ov else []) + [v]
            else:
                cur.append(s)
                curv.append(v)
            prev = sim
        flush()
        return chunks


class QueryExpander:
    def __init__(self):
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

    def expand(self, q):
        words = [w for w in q.lower().split() if w not in self.stop_words]
        aug = []
        for w in words:
            if w.endswith('s') and len(w) > 4:
                aug.append(w[:-1])
            elif not w.endswith('s'):
                aug.append(w + 's')
            if w.endswith('ing'):
                aug.append(w[:-3])
            if w.endswith('ed'):
                aug.append(w[:-2])
        return q + " " + " ".join(aug)


class AdvancedVectorStore:
    """FAISS + BM25 + TF-IDF; optional Pinecone when API key is set."""

    PINECONE_NAMESPACE = "default"

    def __init__(self, embedding_model, pinecone_api_key=None):
        self.emb_model = embedding_model
        self.dim = embedding_model.get_sentence_embedding_dimension()
        self.chunks = []
        self.faiss_index = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.expander = QueryExpander()
        self.pc = None
        self.pinecone_index = None
        self.index_name = PINECONE_INDEX_NAME
        self.uses_integrated_embeddings = False  # Track if index uses integrated embeddings (text-based upsert)
        if pinecone_api_key and (pinecone_api_key or "").strip() and _PINECONE_AVAILABLE:
            try:
                # Same pattern as Pinecone docs: pc = Pinecone(api_key=...); index = pc.Index("name")
                pc = Pinecone(api_key=(pinecone_api_key or "").strip())
                
                # For "ragquery" index, assume it uses integrated embeddings (llama-text-embed-v2)
                # Connect directly without dimension checks
                if self.index_name == "ragquery":
                    self.uses_integrated_embeddings = True
                    print(f"✓ Connecting to '{self.index_name}' (assumed integrated embeddings - llama-text-embed-v2)")
                    print(f"  Will upsert TEXT records - Pinecone generates embeddings automatically.")
                    self.pinecone_index = pc.Index(self.index_name)
                    self.pc = pc
                    print("Vector store: FAISS + Pinecone (cloud).")
                elif not pc.has_index(self.index_name):
                    # New index - create it
                    pc.create_index(
                        name=self.index_name,
                        dimension=self.dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                        deletion_protection="disabled",
                    )
                    time.sleep(2)
                    self.pinecone_index = pc.Index(self.index_name)
                    self.pc = pc
                    print("Vector store: FAISS + Pinecone (cloud).")
                else:
                    # Other existing index - try to detect type
                    try:
                        desc = pc.describe_index(self.index_name)
                        spec = getattr(desc, "spec", None) or (desc.get("spec") if isinstance(desc, dict) else None)
                        if spec:
                            embed = getattr(spec, "embed", None) or (spec.get("embed") if isinstance(spec, dict) else None)
                            if embed:
                                self.uses_integrated_embeddings = True
                                print(f"✓ Index '{self.index_name}' uses integrated embeddings.")
                        self.pinecone_index = pc.Index(self.index_name)
                        self.pc = pc
                        print("Vector store: FAISS + Pinecone (cloud).")
                    except Exception as desc_e:
                        # If describe fails, try to connect anyway
                        print(f"Note: Could not describe '{self.index_name}': {desc_e}. Connecting anyway.")
                        self.pinecone_index = pc.Index(self.index_name)
                        self.pc = pc
                        print("Vector store: FAISS + Pinecone (cloud).")
            except Exception as e:
                print(f"Pinecone connection failed: {e}. Using local FAISS only.")
                import traceback
                print(traceback.format_exc())
                self.pinecone_index = None
                self.pc = None
        if self.pinecone_index is None:
            print("Vector store: local FAISS.")

    def build(self, chunks):
        texts = [c["text"] for c in chunks]
        print(f"Building indexes for {len(texts)} chunks")
        self.chunks = chunks
        E = self.emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(E)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.tfidf_vectorizer = TfidfVectorizer(max_features=6000, stop_words='english', ngram_range=(1, 2), min_df=1, max_df=0.95)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        if self.pinecone_index is not None:
            self._upsert_pinecone(chunks, E)
        print("Indexing complete.")

    def _upsert_pinecone(self, chunks, embeddings, batch_size=100):
        if self.uses_integrated_embeddings:
            # For integrated embeddings (llama-text-embed-v2), upsert TEXT records
            # Pinecone will generate embeddings automatically
            for i in range(0, len(chunks), batch_size):
                records = []
                for j in range(i, min(i + batch_size, len(chunks))):
                    c = chunks[j]
                    meta = c.get("metadata", {})
                    # Ensure UTF-8 encoding for hash calculation
                    text_bytes = c["text"].encode('utf-8', errors='replace')
                    chunk_hash = meta.get("chunk_hash", hashlib.md5(text_bytes).hexdigest()[:16])
                    # Ensure text is UTF-8 encoded string (handle Unicode characters)
                    text_utf8 = c["text"].encode('utf-8', errors='replace').decode('utf-8')
                    records.append({
                        "_id": chunk_hash,
                        "text": text_utf8,  # Full text - Pinecone will embed it
                        "chunk_hash": chunk_hash,
                        "source_file": meta.get("source_file", ""),
                        "idx": int(j)
                    })
                try:
                    self.pinecone_index.upsert_records(self.PINECONE_NAMESPACE, records)
                    print(f"Upserted {len(records)} text records to Pinecone (integrated embeddings)")
                except Exception as e:
                    print(f"Pinecone text upsert batch error: {e}")
        else:
            # For regular indexes, upsert vectors
            for i in range(0, len(chunks), batch_size):
                batch = []
                for j in range(i, min(i + batch_size, len(chunks))):
                    c = chunks[j]
                    meta = c.get("metadata", {})
                    # Ensure UTF-8 encoding for hash calculation
                    text_bytes = c["text"].encode('utf-8', errors='replace')
                    chunk_hash = meta.get("chunk_hash", hashlib.md5(text_bytes).hexdigest()[:16])
                    text_preview = (c["text"][:900] + "…") if len(c["text"]) > 900 else c["text"]
                    batch.append({
                        "id": chunk_hash,
                        "values": embeddings[j].tolist(),
                        "metadata": {"text": text_preview, "chunk_hash": chunk_hash, "source_file": meta.get("source_file", ""), "idx": j},
                    })
                try:
                    self.pinecone_index.upsert(vectors=batch, namespace=self.PINECONE_NAMESPACE)
                    print(f"Upserted {len(batch)} vectors to Pinecone")
                except Exception as e:
                    print(f"Pinecone upsert batch error: {e}")

    def search(self, query, top_k=5, use_expansion=True, rerank=True):
        start = time.time()
        if not self.chunks:
            return [], time.time() - start
        expanded = self.expander.expand(query) if use_expansion else query
        cands = self._retrieve_multi(query, expanded, top_k * 3)
        if rerank and len(cands) > top_k:
            cands = self._rerank(query, cands, top_k * 2)
        final = self._diversify(cands[:top_k * 2], top_k)
        return final[:top_k], time.time() - start

    def _retrieve_multi(self, orig_q, expanded_q, top_k):
        res = {}
        n = len(self.chunks)
        if n == 0:
            return []
        qv = self.emb_model.encode([orig_q], normalize_embeddings=True).astype("float32")
        k = min(top_k * 2, n)
        sims, idxs = self.faiss_index.search(qv, k)
        for i, s in zip(idxs[0], sims[0]):
            if 0 <= i < n:
                res[int(i)] = res.get(int(i), 0) + float(s) * 0.4
        if self.pinecone_index is not None:
            try:
                if self.uses_integrated_embeddings:
                    # For integrated embeddings, query with TEXT (Pinecone generates embedding)
                    pc_res = self.pinecone_index.search(
                        namespace=self.PINECONE_NAMESPACE,
                        query={"top_k": min(top_k * 3, 100), "inputs": {"text": orig_q}}
                    )
                    matches = getattr(pc_res, "result", {}).get("hits", []) or (pc_res.get("result", {}).get("hits") if isinstance(pc_res, dict) else []) or []
                else:
                    # For regular indexes, query with vectors
                    q_list = qv[0].tolist()
                    pc_res = self.pinecone_index.query(vector=q_list, top_k=min(top_k * 3, 100), include_metadata=True, namespace=self.PINECONE_NAMESPACE)
                    matches = getattr(pc_res, "matches", None) or (pc_res.get("matches") if isinstance(pc_res, dict) else []) or []
                
                for match in matches:
                    score_val = float(getattr(match, "score", None) or (match.get("_score") if isinstance(match, dict) else match.get("score", 0)) or 0)
                    meta = getattr(match, "metadata", None) or (match.get("metadata") if isinstance(match, dict) else match.get("fields", {})) or {}
                    idx = meta.get("idx")
                    if idx is not None:
                        try:
                            idx = int(idx)
                        except Exception:
                            idx = None
                    if idx is not None and 0 <= idx < n:
                        res[idx] = res.get(idx, 0) + score_val * 0.4
                    else:
                        chunk_hash = meta.get("chunk_hash")
                        for i, ch in enumerate(self.chunks):
                            if (ch.get("metadata") or {}).get("chunk_hash") == chunk_hash:
                                res[i] = res.get(i, 0) + score_val * 0.4
                                break
            except Exception as e:
                print(f"Pinecone query error: {e}")
        bm = self.bm25.get_scores(expanded_q.lower().split())
        mbm = max(bm) if len(bm) else 1
        for i in np.argsort(bm)[::-1][:k]:
            res[i] = res.get(i, 0) + (bm[i] / mbm if mbm > 0 else 0) * 0.3
        qtf = self.tfidf_vectorizer.transform([expanded_q])
        tf = cosine_similarity(qtf, self.tfidf_matrix).flatten()
        for i in np.argsort(tf)[::-1][:k]:
            res[i] = res.get(i, 0) + float(tf[i]) * 0.3
        ranked = sorted(res.items(), key=lambda x: x[1], reverse=True)
        out = []
        for idx, score in ranked[:top_k * 2]:
            if idx >= n:
                continue
            t = self.chunks[idx]["text"]
            meta = self.chunks[idx]["metadata"].copy()
            score += DomainHeuristics.domain_score(t)
            out.append({"text": t, "metadata": meta, "relevance": round(score, 4), "chunk_idx": idx, "source_file": meta.get("source_file", "unknown")})
        out.sort(key=lambda x: x["relevance"], reverse=True)
        return out

    def _rerank(self, query, cands, top_k):
        q_terms = set(query.lower().split())
        def score(c):
            text = c["text"].lower()
            txt_terms = set(text.split())
            overlap = len(q_terms & txt_terms) / max(1, len(q_terms))
            pos_scores = [1 - (text.find(t) / max(1, len(text))) for t in q_terms if t in text]
            pos = np.mean(pos_scores) if pos_scores else 0
            return c["relevance"] * 0.6 + overlap * 0.25 + pos * 0.15 + DomainHeuristics.domain_score(text) * 0.5
        return sorted(cands, key=score, reverse=True)[:top_k]

    def _diversify(self, cands, top_k):
        if len(cands) <= top_k:
            return cands
        sel = [cands[0]]
        rem = cands[1:]
        while len(sel) < top_k and rem:
            best = None
            best_score = -1
            for c in rem:
                sims = []
                for s in sel:
                    w1 = set(c["text"].lower().split())
                    w2 = set(s["text"].lower().split())
                    j = len(w1 & w2) / max(1, len(w1 | w2))
                    sims.append(j)
                div = 1 - max(sims) if sims else 1
                sc = c["relevance"] * 0.7 + div * 0.3
                if sc > best_score:
                    best_score = sc
                    best = c
            sel.append(best)
            rem.remove(best)
        return sel


class EnhancedQAModel:
    def __init__(self):
        m = "deepset/roberta-base-squad2"
        self.pipe = None
        self._tok = None
        self._model = None
        try:
            self.pipe = pipeline("question-answering", model=m, tokenizer=m, device=0 if torch.cuda.is_available() else -1)
        except Exception:
            # transformers >= 5 removed the "question-answering" pipeline task;
            # fall back to direct AutoModelForQuestionAnswering inference.
            try:
                from transformers import AutoTokenizer, AutoModelForQuestionAnswering
                self._tok = AutoTokenizer.from_pretrained(m)
                self._model = AutoModelForQuestionAnswering.from_pretrained(m)
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                print("Extractive QA: using direct AutoModelForQuestionAnswering (pipeline task unavailable).")
            except Exception as e:
                print(f"Extractive QA model unavailable (startup will continue): {e}")

    def _qa_direct(self, question, context):
        inputs = self._tok(question, context, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        start_probs = torch.softmax(out.start_logits[0], dim=-1)
        end_probs = torch.softmax(out.end_logits[0], dim=-1)
        s_idx = int(start_probs.argmax())
        e_idx = int(end_probs.argmax())
        if e_idx < s_idx or e_idx - s_idx > 50:
            return {"answer": "", "score": 0.0, "start": -1, "end": -1}
        score = float(start_probs[s_idx] * end_probs[e_idx])
        span_ids = inputs["input_ids"][0][s_idx : e_idx + 1]
        answer = self._tok.decode(span_ids, skip_special_tokens=True).strip()
        return {"answer": answer, "score": score, "start": -1, "end": -1}

    def answer(self, question, hits):
        start = time.time()
        if self.pipe is None and self._model is None:
            return {"found": False, "confidence": 0.0, "processing_time": time.time() - start}
        if not hits:
            return {"found": False, "confidence": 0.0, "processing_time": 0.0}
        best = None
        score = 0.0
        for h in hits[:3]:
            try:
                context = h["text"][:4000]
                out = self.pipe({"question": question, "context": context}) if self.pipe else self._qa_direct(question, context)
                ans = (out.get("answer") or "").strip()
                sc = float(out.get("score", 0.0))
                if sc > score and ans and len(ans.split()) >= 2:
                    proc = self._process_answer(ans, context, out.get("start", -1), out.get("end", -1))
                    if proc:
                        best = {"answer": proc, "score": sc}
                        score = sc
            except Exception:
                pass
        t = time.time() - start
        if not best or score < 0.3:
            return {"found": False, "confidence": score, "processing_time": t}
        return {"found": True, "answer": best["answer"], "score": score, "confidence": min(1.0, score * 1.2), "processing_time": t}

    def _process_answer(self, ans, ctx, start, end):
        if start == -1 or end == -1:
            start = ctx.lower().find(ans.lower())
            if start == -1:
                return None
            end = start + len(ans)
        sent_start = start
        while sent_start > 0 and ctx[sent_start - 1] not in '.!?\n':
            sent_start -= 1
        if sent_start > 0 and ctx[sent_start - 1] in '.!?':
            sent_start += 1
        sent_end = end
        while sent_end < len(ctx) and ctx[sent_end] not in '.!?\n':
            sent_end += 1
        if sent_end < len(ctx) and ctx[sent_end] in '.!?':
            sent_end += 1
        sentence = re.sub(r'\s+', ' ', ctx[sent_start:sent_end].strip())
        if ans.lower() not in sentence.lower():
            return ans.strip() + "." if not sentence.endswith(('.', '!', '?')) else ans.strip()
        if not sentence.endswith(('.', '!', '?')) and len(sentence.split()) > 3:
            sentence += "."
        return sentence
