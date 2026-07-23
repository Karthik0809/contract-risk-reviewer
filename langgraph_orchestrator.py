from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import START, END, StateGraph
    from langchain_core.prompts import PromptTemplate
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


class GraphState(TypedDict, total=False):
    question: str
    history: List[Any]
    k: int
    confidence_threshold: float
    use_expansion: bool
    enable_reranking: bool
    show_debug: bool
    max_answer_tokens: int
    route: str
    hits: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    qa_time: float
    answer: str
    method: str
    confidence: float
    max_relevance: float
    error: str
    question_for_generation: str


class LangGraphRAGOrchestrator:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.available = LANGGRAPH_AVAILABLE
        self.graph = None
        if self.available:
            self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("route", self._route_question)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("summary", self._summary)
        workflow.add_node("compare", self._compare)
        workflow.add_node("refine", self._refine)
        workflow.add_node("extractive", self._extractive)
        workflow.add_node("generative", self._generative_with_langchain_prompt)
        workflow.add_node("sentence_fallback", self._sentence_fallback)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route",
            self._route_next,
            {
                "summary": "retrieve",
                "compare": "compare",
                "refine": "retrieve",
                "qa": "retrieve",
                "error": "finalize",
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self._retrieve_next,
            {
                "summary": "summary",
                "refine": "refine",
                "qa": "extractive",
                "error": "finalize",
            },
        )
        workflow.add_conditional_edges(
            "extractive",
            self._extractive_next,
            {
                "done": "finalize",
                "generative": "generative",
            },
        )
        workflow.add_conditional_edges(
            "generative",
            self._generative_next,
            {
                "done": "finalize",
                "fallback": "sentence_fallback",
            },
        )
        workflow.add_edge("summary", "finalize")
        workflow.add_edge("compare", "finalize")
        workflow.add_edge("refine", "finalize")
        workflow.add_edge("sentence_fallback", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def answer(
        self,
        question,
        history,
        k=5,
        confidence_threshold=0.3,
        use_expansion=True,
        max_answer_tokens=0,
        enable_reranking=True,
        show_debug=True,
    ):
        if not self.available or self.graph is None:
            raise RuntimeError("LangGraph is not available")

        if not question or not question.strip():
            return "", history
        if not self.chatbot.vs.chunks:
            return "", history + [(question, "Please upload and process documents first (left panel).")]

        response_start = time.time()
        state: GraphState = {
            "question": question,
            "history": history,
            "k": int(k),
            "confidence_threshold": float(confidence_threshold),
            "use_expansion": bool(use_expansion),
            "enable_reranking": bool(enable_reranking),
            "show_debug": bool(show_debug),
            "max_answer_tokens": int(max_answer_tokens or 0),
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "qa_time": 0.0,
            "hits": [],
            "answer": "",
            "method": "langgraph",
            "confidence": 0.0,
            "max_relevance": 0.0,
            "question_for_generation": question,
        }

        out = self.graph.invoke(state)
        total_time = time.time() - response_start

        if out.get("error"):
            self.chatbot.metrics.log_query(total_time, 0.0, 0.0, 0.0, 0.0, 0, "error", 0.0, False)
            return "", history + [(question, f"Error processing question: {out['error']}")]

        final = out.get("answer") or "The requested information is not available in the uploaded documents."
        method = out.get("method", "langgraph")
        conf = float(out.get("confidence", 0.0))
        retrieval_time = float(out.get("retrieval_time", 0.0))
        generation_time = float(out.get("generation_time", 0.0))
        qa_time = float(out.get("qa_time", 0.0))
        max_relevance = float(out.get("max_relevance", 0.0))
        hits = out.get("hits", [])

        has_relevant = len(hits) > 0 and max_relevance > 0.3
        self.chatbot.metrics.log_query(
            response_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            extraction_time=qa_time,
            max_relevance=max_relevance,
            answer_length=len(final),
            method=method,
            confidence=conf,
            has_relevant_results=has_relevant,
        )

        if show_debug:
            debug_section = self.chatbot.debug_info.format_debug_info(
                hits, method, conf, total_time, retrieval_time, generation_time, question
            )
            final = final + debug_section

        return "", history + [(question, final)]

    def _route_question(self, state: GraphState) -> GraphState:
        question = (state.get("question") or "").lower()
        if not question.strip():
            return {"route": "error", "error": "Question is empty"}

        summary_phrases = (
            "what does the document say",
            "summarize",
            "summary",
            "what is this about",
            "main topics",
            "key points",
            "tell me about this document",
        )
        if any(p in question for p in summary_phrases):
            return {"route": "summary"}
        if "compare" in question and len(self.chatbot.processed) >= 2:
            return {"route": "compare"}
        if "explain more" in question or "more detail" in question:
            return {"route": "refine"}
        return {"route": "qa"}

    def _route_next(self, state: GraphState) -> str:
        return state.get("route", "qa")

    def _retrieve(self, state: GraphState) -> GraphState:
        try:
            top_k = state["k"]
            # Summary questions need broader context than factoid QA.
            if state.get("route") == "summary":
                top_k = max(int(top_k), 8)
            else:
                # Factoid QA benefits from slightly wider recall.
                top_k = max(int(top_k), 8)

            search_query = state["question"]
            q_low = (search_query or "").lower()
            # Lightweight domain-agnostic expansion for factoid questions.
            extra_terms: List[str] = []
            if "where" in q_low:
                extra_terms.extend(["location", "place", "institution", "at", "in"])
            if "when" in q_low:
                extra_terms.extend(["date", "year", "timeline", "period"])
            if "who" in q_low:
                extra_terms.extend(["person", "name", "role"])
            if "program" in q_low:
                extra_terms.extend(["program", "course", "track", "major"])
            if "degree" in q_low:
                extra_terms.extend(["degree", "qualification", "education"])
            if extra_terms:
                search_query = f"{search_query} {' '.join(extra_terms)}"
            hits, rt = self.chatbot.vs.search(
                search_query,
                top_k=top_k,
                use_expansion=state["use_expansion"],
                rerank=state["enable_reranking"],
            )
            # Final domain-agnostic lexical re-rank pass so shown chunks match the question better.
            hits = self._post_rank_hits_by_query_relevance(state.get("question", ""), hits)
            max_rel = max((h.get("relevance", 0.0) for h in hits), default=0.0)
            return {"hits": hits, "retrieval_time": rt, "max_relevance": max_rel}
        except Exception as e:
            return {"route": "error", "error": str(e)}

    def _post_rank_hits_by_query_relevance(self, question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank retrieved chunks by lexical overlap + semantic score.

        This keeps retrieval generic for all document types while improving alignment
        between asked question and displayed supporting chunks.
        """
        if not hits:
            return hits

        q = (question or "").lower()
        q_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2]
        stop = {
            "the", "and", "for", "with", "from", "this", "that", "what", "where", "when",
            "which", "who", "how", "does", "did", "into", "about", "more", "detailed"
        }
        q_tokens = [t for t in q_tokens if t not in stop]

        # Generic phrase hints for structured facts.
        phrase_hints: List[str] = []
        if "where" in q:
            phrase_hints.extend([" at ", " in ", "university", "college", "location", "based"])
        if any(k in q for k in ("gpa", "cgpa", "grade", "score")):
            phrase_hints.extend(["gpa", "cgpa", "/4", "/10", "grade"])
        if any(k in q for k in ("master", "bachelor", "degree", "program", "course")):
            phrase_hints.extend(["master", "bachelor", "degree", "program", "course"])

        scored = []
        for h in hits:
            text = re.sub(r"\s+", " ", (h.get("text") or "")).lower()
            base = float(h.get("relevance", 0.0))
            if not q_tokens:
                scored.append((base, h))
                continue

            overlap = sum(1 for t in q_tokens if t in text)
            phrase = sum(1 for p in phrase_hints if p in text)
            # Favor chunks that explicitly match question terms while retaining vector semantics.
            score = (0.65 * base) + (0.25 * overlap) + (0.10 * phrase)
            scored.append((score, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored]

    def _retrieve_next(self, state: GraphState) -> str:
        if not state.get("hits"):
            return "error"
        route = state.get("route", "qa")
        if route == "summary":
            return "summary"
        if route == "refine":
            return "refine"
        return "qa"

    def _summary(self, state: GraphState) -> GraphState:
        try:
            hits = state.get("hits", [])
            if not hits:
                return {"answer": "The requested information is not available in the uploaded documents."}

            question = (state.get("question") or "").lower()
            intent = self._summary_intent(question)

            # Primary: dedicated summary method
            summary, st = self.chatbot.gen.generate_summary(hits)
            if summary and len(summary.split()) >= 12 and not self._looks_truncated(summary):
                return {
                    "answer": self.chatbot._clean_answer(summary),
                    "method": f"langgraph_{intent}",
                    "confidence": 0.75,
                    "generation_time": st,
                }

            # Fallback 1: robust summary prompt over wider context (raw text + retrieved snippets).
            style_hint = self._summary_style_hint(intent)
            output_hint = self._summary_output_hint(intent)

            summary_prompt = PromptTemplate.from_template(
                "You are a helpful document assistant.\n"
                "Based ONLY on the provided context, answer the user request accurately.\n"
                "Do not mention chunk ids, retrieval metadata, or file names.\n"
                "Do not output incomplete bullet fragments.\n"
                "{style_hint}\n\n"
                "Context:\n{context}\n\nSummary:"
            )
            # Raw document text is more stable for broad summary prompts than top-k retrieval only.
            raw_parts = []
            for _, txt in list(self.chatbot.raw_texts.items())[:3]:
                cleaned = re.sub(r"\s+", " ", txt).strip()
                if cleaned:
                    raw_parts.append(cleaned[:3500])
            raw_ctx = "\n\n".join(raw_parts)

            retrieved_ctx = "\n\n".join(re.sub(r"\s+", " ", h["text"]).strip()[:1200] for h in hits[:8])

            ctx = (raw_ctx + "\n\n" + retrieved_ctx).strip()
            rendered = summary_prompt.format(context=ctx, style_hint=style_hint)
            gen_txt, gen_t = self._generate_text(rendered)
            if gen_txt and len(gen_txt.split()) >= 12 and not gen_txt.lower().startswith(
                ("insufficient", "error", "not stated")
            ) and not self._looks_truncated(gen_txt):
                gen_txt = self._normalize_summary_shape(gen_txt, intent, output_hint)
                return {
                    "answer": self.chatbot._clean_answer(gen_txt),
                    "method": f"langgraph_{intent}_fallback",
                    "confidence": 0.7,
                    "generation_time": st + gen_t,
                }

            # Fallback 2: deterministic multi-point summary from top chunks.
            # This avoids repeating a single sentence for all summary prompts.
            structured = self._structured_summary_from_hits(hits, intent=intent)
            if structured:
                return {
                    "answer": structured,
                    "method": f"langgraph_{intent}_structured_fallback",
                    "confidence": 0.55,
                    "generation_time": st + gen_t,
                }

            return {"answer": "The requested information is not available in the uploaded documents."}
        except Exception as e:
            return {"error": str(e)}

    def _summary_intent(self, question: str) -> str:
        if any(x in question for x in ("main topic", "main topics", "subjects", "subject covered", "what topics")):
            return "main_topics"
        if any(x in question for x in ("key point", "main highlight", "highlights")):
            return "key_points"
        return "summary"

    def _summary_style_hint(self, intent: str) -> str:
        if intent == "main_topics":
            return "Return the major topics/subject areas covered."
        if intent == "key_points":
            return "Return key points and major highlights."
        return "Return a concise overall summary."

    def _summary_output_hint(self, intent: str) -> str:
        if intent == "main_topics":
            return "Use 4-6 short bullets named as topics."
        if intent == "key_points":
            return "Use 4-6 bullets with concrete highlights."
        return "Use a short paragraph followed by 3-5 bullets."

    def _normalize_summary_shape(self, text: str, intent: str, output_hint: str) -> str:
        # Lightweight formatting guidance if model returns plain block text.
        cleaned = text.strip()
        if "- " in cleaned or "\n" in cleaned:
            return cleaned
        if intent == "summary":
            return cleaned
        # For key_points/topics, force scan-friendly bullets from sentence splits.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip().split()) >= 5]
        if not sentences:
            return cleaned
        header = "Key points:" if intent == "key_points" else "Main topics:"
        bullets = "\n".join(f"- {s}" for s in sentences[:5])
        return f"{header}\n{bullets}"

    def _looks_truncated(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        if t.endswith(("vs.", "and", "or", "with", "using", "through", "by", "to", "of")):
            return True
        if t.count("(") != t.count(")"):
            return True
        return False

    def _normalize_factual_answer(self, question: str, answer: str, hits: List[Dict[str, Any]]) -> str:
        """Convert extractive spans into concise direct answers for factoid queries."""
        q = (question or "").lower().strip()
        a = (answer or "").strip()
        if not a:
            return a

        corpus = " ".join([a] + [h.get("text", "") for h in hits[:3]])
        corpus = re.sub(r"\s+", " ", corpus)

        # WHERE intent -> prefer concise location/entity extraction.
        if "where" in q:
            # Common institution/location patterns across domains.
            patterns = [
                r"\b(University of [A-Z][A-Za-z&.\- ]+)\b",
                r"\b(University at [A-Z][A-Za-z&.\- ]+)\b",
                r"\b([A-Z][A-Za-z&.\- ]+ University)\b",
                r"\b(at [A-Z][A-Za-z&.\- ]{3,60})\b",
                r"\b(in [A-Z][A-Za-z&.\- ]{3,60})\b",
            ]
            for pat in patterns:
                m = re.search(pat, corpus)
                if m:
                    ent = m.group(1).strip(" ,.")
                    if ent.lower().startswith(("at ", "in ")):
                        return f"The document indicates it was {ent}."
                    return f"The document indicates it was at {ent}."

        # Program-like intent -> concise program phrase if present.
        if any(k in q for k in ("program", "course", "track", "major", "admission")):
            prog_match = re.search(
                r"\b((?:Ph\.?D\.?|Master'?s|Bachelor'?s)\s+(?:program|degree)\s+in\s+[A-Za-z &\-]+)\b",
                corpus,
                flags=re.I,
            )
            if prog_match:
                prog = re.sub(r"\s+", " ", prog_match.group(1)).strip(" ,.")
                return f"The document indicates the program is {prog}."

        return a

    def _direct_fact_lookup(self, question: str) -> str:
        """Fast fact lookup on full extracted text (generic, domain-agnostic)."""
        q = (question or "").lower().strip()
        if not q:
            return ""

        texts = []
        for _, txt in (self.chatbot.raw_texts or {}).items():
            t = re.sub(r"\s+", " ", txt or "").strip()
            if t:
                texts.append(t)
        if not texts:
            return ""

        corpus = " ".join(texts)

        asks_name = any(x in q for x in ("name", "who is", "what is his name", "what is her name", "person name"))
        asks_gpa = any(x in q for x in ("gpa", "cgpa", "grade point"))
        asks_masters = any(x in q for x in ("master", "masters", "master's"))

        name_value = None
        if asks_name:
            m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", corpus)
            if m:
                name_value = m.group(1).strip()

        gpa_value = None
        for pat in [
            r"\b(?:CGPA|GPA)\s*[:\-]?\s*([0-9](?:\.[0-9]{1,2})?\s*/\s*[0-9](?:\.[0-9])?)\b",
            r"\b([0-9](?:\.[0-9]{1,2})?\s*/\s*4\.0)\b",
            r"\b([0-9](?:\.[0-9]{1,2})?\s*/\s*10(?:\.0)?)\b",
        ]:
            m = re.search(pat, corpus, flags=re.I)
            if m:
                gpa_value = m.group(1).replace(" ", "")
                break

        masters_inst = None
        if asks_masters:
            m = re.search(
                r"(?:master'?s(?:\s+degree|\s+program)?(?:\s+at|\s+from)?\s+)([A-Z][A-Za-z&.\- ]{3,80})",
                corpus,
                flags=re.I,
            )
            if m:
                masters_inst = m.group(1).strip(" ,.")
            else:
                m2 = re.search(r"\b(University (?:at|of) [A-Z][A-Za-z&.\- ]+)\b", corpus)
                if m2:
                    masters_inst = m2.group(1).strip(" ,.")

        parts = []
        if asks_name and name_value:
            parts.append(f"The name in the document is {name_value}.")
        if asks_gpa:
            if gpa_value:
                if asks_masters and masters_inst:
                    parts.append(f"The master's GPA/CGPA is {gpa_value} at {masters_inst}.")
                else:
                    parts.append(f"The GPA/CGPA in the document is {gpa_value}.")
            else:
                parts.append("The GPA/CGPA is not stated in the uploaded documents.")

        return " ".join(parts).strip()

    def _structured_summary_from_hits(self, hits: List[Dict[str, Any]], intent: str = "summary") -> str:
        """Build intent-aware concise summary from multiple chunks without LLM."""
        # Keep order while deduplicating similar candidate lines.
        candidates: "OrderedDict[str, None]" = OrderedDict()
        for h in hits[:8]:
            text = re.sub(r"\s+", " ", (h.get("text") or "")).strip()
            if not text:
                continue
            # Split by common bullet separators and sentence boundaries.
            parts = re.split(r"[•\n]|(?<=[.!?])\s+", text)
            for p in parts:
                p = p.strip(" -\t")
                if len(p.split()) < 7:
                    continue
                low = p.lower()
                # Skip noisy metadata-like fragments.
                if any(x in low for x in ["source:", "similarity:", "chunk", "method used"]):
                    continue
                # Skip role/date header fragments that read badly as standalone summary points.
                if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", low) and len(p.split()) <= 14:
                    continue
                if self._looks_truncated(p):
                    continue
                # Normalize key for dedupe.
                key = re.sub(r"[^a-z0-9 ]+", "", low)
                key = re.sub(r"\s+", " ", key).strip()
                if key and key not in candidates:
                    candidates[key] = p

        lines = list(candidates.values())
        if not lines:
            return ""

        # Build 4-6 point summary from distinct lines.
        picked = lines[:5]
        if intent == "main_topics":
            header = "Main topics and subjects covered:"
        elif intent == "key_points":
            header = "Key points and highlights:"
        else:
            header = "Summary of the document:"
        bullets = []
        for line in picked:
            clean = self.chatbot._clean_answer(line)
            if clean and not clean.endswith("."):
                clean += "."
            bullets.append(f"- {clean}")
        return header + "\n" + "\n".join(bullets)

    def _compare(self, state: GraphState) -> GraphState:
        try:
            if len(self.chatbot.processed) < 2:
                return {"answer": "Need at least 2 documents to compare.", "method": "langgraph_compare"}
            doc1 = self.chatbot.processed[0]["file"]
            doc2 = self.chatbot.processed[1]["file"]
            answer = self.chatbot.compare_documents(doc1, doc2, state["question"])
            return {"answer": answer, "method": "langgraph_compare", "confidence": 0.75}
        except Exception as e:
            return {"error": str(e)}

    def _refine(self, state: GraphState) -> GraphState:
        try:
            hits = state.get("hits", [])
            if not hits:
                return {"answer": "No additional context found to refine the answer."}
            q = state.get("question", "")
            prompt = PromptTemplate.from_template(
                "Question: {question}\n\nContext:\n{context}\n\n"
                "Provide a deeper explanation with examples and important details."
            )
            ctx = "\n\n".join(re.sub(r"\s+", " ", h["text"]).strip()[:1500] for h in hits[:4])
            rendered = prompt.format(question=q, context=ctx)
            answer, gt = self._generate_text(rendered)
            if answer:
                return {
                    "answer": self.chatbot._clean_answer(answer),
                    "method": "langgraph_refine",
                    "confidence": 0.75,
                    "generation_time": gt,
                }
            return {"answer": "Unable to generate a refined explanation at this time."}
        except Exception as e:
            return {"error": str(e)}

    def _extractive(self, state: GraphState) -> GraphState:
        fact = self._direct_fact_lookup(state.get("question", ""))
        if fact:
            return {
                "answer": self.chatbot._clean_answer(fact),
                "method": "langgraph_fact_lookup",
                "confidence": 0.78,
                "qa_time": 0.0,
            }

        qa = self.chatbot.qa.answer(state["question"], state["hits"])
        conf = qa.get("confidence", 0.0)
        # Avoid over-rejecting otherwise valid factual spans.
        configured = float(state.get("confidence_threshold", 0.3))
        threshold = max(0.15, min(0.3, configured))
        if qa.get("found") and conf >= threshold:
            raw_answer = qa.get("answer", "")
            normalized = self._normalize_factual_answer(
                question=state.get("question", ""),
                answer=raw_answer,
                hits=state.get("hits", []),
            )
            return {
                "answer": self.chatbot._clean_answer(normalized),
                "method": "langgraph_extractive_qa",
                "confidence": conf,
                "qa_time": qa.get("processing_time", 0.0),
            }
        return {"qa_time": qa.get("processing_time", 0.0)}

    def _extractive_next(self, state: GraphState) -> str:
        return "done" if state.get("answer") else "generative"

    def _generative_with_langchain_prompt(self, state: GraphState) -> GraphState:
        hits = state.get("hits", [])
        if not hits:
            return {}
        question = state.get("question_for_generation") or state.get("question", "")
        prompt = PromptTemplate.from_template(
            "You are a helpful document assistant.\n"
            "Answer only from the context below in clear plain English.\n"
            "If not present, reply: Not stated in the provided documents.\n\n"
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        context = "\n\n".join(re.sub(r"\s+", " ", h["text"]).strip()[:1800] for h in hits[:4])
        rendered = prompt.format(question=question, context=context)

        answer, gt = self._generate_text(rendered)
        if answer and len(answer.split()) >= 3 and not answer.lower().startswith(("insufficient", "error", "not stated")):
            return {
                "answer": self.chatbot._clean_answer(answer),
                "method": "langgraph_generative",
                "confidence": 0.75 if self.chatbot.gen.api_key_set else 0.3,
                "generation_time": gt,
            }
        return {"generation_time": gt}

    def _generative_next(self, state: GraphState) -> str:
        return "done" if state.get("answer") else "fallback"

    def _sentence_fallback(self, state: GraphState) -> GraphState:
        ans = self.chatbot._extract_relevant_sentences(state["question"], state.get("hits", []))
        conf = 0.5 if ans != "Information not found in the provided documents." else 0.2
        return {"answer": ans, "method": "langgraph_sentence_extraction", "confidence": conf}

    def _finalize(self, state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        if not answer or answer.lower().startswith("information not found"):
            answer = "The requested information is not available in the uploaded documents."
        return {"answer": answer}

    def _generate_text(self, rendered_prompt: str):
        start = time.time()
        if not self.chatbot.gen.api_key_set or not self.chatbot.gen.model:
            return "Generative answer unavailable: set a valid Google API key in the left panel.", time.time() - start
        try:
            response = self.chatbot.gen.model.generate_content(rendered_prompt)
            txt = (response.text or "").strip()
            txt = self.chatbot.gen._clean_response(txt)
            return txt, time.time() - start
        except Exception as e:
            return f"Error generating response: {e}", time.time() - start
