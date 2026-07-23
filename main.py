# Main application file for Enhanced RAG Chatbot
# -*- coding: utf-8 -*-
import os
import re
import time
import sys
import io
import torch
from sentence_transformers import SentenceTransformer

# Fix Windows encoding issues - force UTF-8 for console output
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from config import DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_API_KEY, DEFAULT_PINECONE_API_KEY
from config import USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY
from utils import PerformanceMetrics, ResponseDebugInfo, ErrorRecovery, safe_sent_tokenize
from document_processor import EnhancedDocumentProcessor
from rag_engine import HierarchicalChunker, AdvancedVectorStore, EnhancedQAModel, OpenAIEmbedder, HashingEmbedder
from llm_interface import GeminiGenerator, SpeechProcessor, BilingualProcessor
from export_manager import ExportManager
from langgraph_orchestrator import LangGraphRAGOrchestrator

class EnhancedRAGChatbot:
    def _init_embedding_model(self, device: str):
        if USE_OPENAI_EMBEDDINGS:
            if OPENAI_API_KEY and (OPENAI_API_KEY or "").strip():
                return OpenAIEmbedder(api_key=OPENAI_API_KEY), "openai/text-embedding-3-large"
            print("WARNING: USE_OPENAI_EMBEDDINGS=True but OPENAI_API_KEY is not set.")
        try:
            emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            return emb, "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as e:
            print(f"Embedding model download failed; using local hashing fallback: {e}")
            return HashingEmbedder(dim=384), "local-hashing-embedder"

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {device.upper()}")
        self.embed, self.embedding_model_name = self._init_embedding_model(device)
        print(f"Embedding backend: {self.embedding_model_name}")
        self.doc = EnhancedDocumentProcessor()
        self.chunker = HierarchicalChunker(self.embed)
        # Vector store: FAISS + optional Pinecone when API key is set
        self.vs = AdvancedVectorStore(self.embed, pinecone_api_key=DEFAULT_PINECONE_API_KEY)
        self._last_chunks = []
        self.qa = EnhancedQAModel()
        self.gen = GeminiGenerator()
        # Auto-configure Gemini with pre-defined key
        if DEFAULT_GEMINI_API_KEY and len(DEFAULT_GEMINI_API_KEY.strip()) >= 20:
            self.gen.configure(DEFAULT_GEMINI_API_KEY.strip())
            print("Gemini configured with pre-defined API key")
        self.processed = []
        self.total_chunks = 0
        self.document_stats = {}
        self.raw_texts = {}  # filename -> raw extracted text
        self.document_sections = {}  # filename -> sections
        self.components_table = {}
        
        # Speech and Language Processing
        self.speech_processor = SpeechProcessor()
        self.bilingual_processor = BilingualProcessor()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.debug_info = ResponseDebugInfo()
        
        # New enhanced features
        self.export_manager = ExportManager()
        self.error_recovery = ErrorRecovery()
        self.langgraph_orchestrator = LangGraphRAGOrchestrator(self)
        if self.langgraph_orchestrator.available:
            print("LangGraph orchestration enabled (LangChain prompts + graph routing).")
        else:
            print("LangGraph not available; using legacy RAG flow.")

    def _set_components_table(self):
        """Set up components table for tracking document components"""
        self.components_table = {
            "embedding_model": getattr(self, "embedding_model_name", "unknown"),
            "chunker": "HierarchicalChunker",
            "vector_store": "AdvancedVectorStore",
            "qa_model": "EnhancedQAModel",
            "generator": "GeminiGenerator",
            "orchestration": "LangGraphRAGOrchestrator" if self.langgraph_orchestrator.available else "Legacy pipeline",
            "total_chunks": self.total_chunks,
            "document_stats": self.document_stats
        }

    # --- Configure API key + model from UI
    def set_gemini(self, api_key: str):
        print(f"set_gemini called with API key: {'Yes' if api_key else 'No'}")
        print(f"API key length: {len(api_key) if api_key else 0}")
        print(f"Preferred model: {DEFAULT_GEMINI_MODEL}")
        
        if not api_key or not api_key.strip():
            status_msg = "No API key provided."
            gemini_status = f"**Gemini Configuration Failed**\nPlease enter your Google Gemini API key."
            print(f"Configuration failed: {status_msg}")
            return status_msg, gemini_status
        
        api_key_clean = api_key.strip()
        
        # Basic validation - Gemini API keys usually start with "AIza" or are longer than 20 chars
        if len(api_key_clean) < 20:
            status_msg = "API key appears to be too short. Please check your API key."
            gemini_status = f"**Gemini Configuration Failed**\nAPI key appears invalid.\nPlease verify you copied the complete API key."
            print(f"Configuration failed: {status_msg}")
            return status_msg, gemini_status
        
        try:
            ok = self.gen.configure(api_key_clean)
            
            if ok and self.gen.api_key_set:
                status_msg = f"Gemini configured successfully with model: {self.gen.model_name}"
                gemini_status = f"**Gemini Ready!**\nModel: {self.gen.model_name}\nStatus: Active"
                print(f"Configuration successful: {status_msg}")
                return status_msg, gemini_status
            else:
                status_msg = f"Failed to configure Gemini. The API key may be invalid."
                gemini_status = f"**Gemini Configuration Failed**\nPlease check your API key.\n\nCommon issues:\n- API key is incorrect or expired\n- API key doesn't have access to Gemini models\n- API key has extra spaces or characters\n\nPlease verify your API key at: https://makersuite.google.com/app/apikey"
                print(f"Configuration failed: {status_msg}")
                return status_msg, gemini_status
        except Exception as e:
            error_msg = str(e)
            print(f"Exception during Gemini configuration: {error_msg}")
            if "API key" in error_msg.lower() or "401" in error_msg or "Unauthorized" in error_msg or "invalid" in error_msg.lower():
                status_msg = "Invalid API key. Please check your Google Gemini API key."
                gemini_status = f"**Gemini Configuration Failed**\nInvalid API key.\n\nPlease:\n1. Verify your API key at https://makersuite.google.com/app/apikey\n2. Make sure you copied the complete key\n3. Ensure the key has access to Gemini models"
            else:
                status_msg = f"Configuration error: {error_msg[:100]}"
                gemini_status = f"**Gemini Configuration Failed**\nError: {error_msg[:100]}"
            print(f"Configuration error: {status_msg}")
            return status_msg, gemini_status
    
    def set_pinecone(self, api_key: str):
        """Set Pinecone API key and use cloud index for search. Empty key = local only."""
        key = (api_key or "").strip()
        try:
            self.vs = AdvancedVectorStore(self.embed, pinecone_api_key=key if key else None)
            if getattr(self, "_last_chunks", None):
                self.vs.build(self._last_chunks)
                self.total_chunks = len(self._last_chunks)
            if self.vs.pinecone_index is not None:
                status_msg = "Pinecone connected. Vector store is using FAISS + Pinecone."
                pinecone_status = "**Vector store**\nConnected to Pinecone (cloud)."
            else:
                status_msg = "Using local storage (FAISS)."
                pinecone_status = "**Vector store**\nUsing local storage (FAISS)."
            return status_msg, pinecone_status
        except Exception as e:
            status_msg = f"Pinecone setup failed: {e}"
            pinecone_status = f"**Vector store**\nError: {str(e)[:80]}"
            return status_msg, pinecone_status

    def process_documents(self, file_paths):
        if not file_paths: 
            return "No files provided. Upload PDF(s) to begin."
        msgs = ["Starting Enhanced Document Processing Pipeline"]
        all_chunks = []
        stats = {"total_pages":0, "total_text_length":0, "extraction_methods": {}}

        for path in file_paths:
            try:
                extraction_start = time.time()
                fname = os.path.basename(path)
                msgs.append(f"Processing {fname}")
                text, meta = self.doc.extract_text(path)
                extraction_time = time.time() - extraction_start

                if meta.get("extraction_method")=="failed":
                    msgs.append(f"Failed: {text}")
                    continue
                
                stats["total_pages"] += meta.get("total_pages",0)
                stats["total_text_length"] += len(text)
                if meta["extraction_method"] not in stats["extraction_methods"]:
                    stats["extraction_methods"][meta["extraction_method"]] = 0
                stats["extraction_methods"][meta["extraction_method"]] += 1
                
                # Store document sections
                if "sections" in meta:
                    self.document_sections[fname] = meta["sections"]
                    section_summary = self.doc.segmenter.get_section_summary(meta["sections"])
                    msgs.append(f"Document sections detected: {len(meta['sections'])} sections")
                
                chunks = self.chunker.split(text, meta)
                all_chunks.extend(chunks)
                dist = {}
                for c in chunks:
                    chunk_type = c["metadata"].get("chunk_type","medium")
                    dist[chunk_type] = dist.get(chunk_type, 0) + 1
                
                self.processed.append({"file": meta["file"], "pages": meta.get("total_pages",0), "chunks": len(chunks),
                                       "chunk_distribution": dist, "extraction_method": meta["extraction_method"],
                                       "structure": meta.get("structure", {}), "extraction_time": extraction_time,
                                       "sections": meta.get("sections", {})})
                self.raw_texts[fname] = text
                msgs.append(f"Created {len(chunks)} chunks (extraction: {extraction_time:.2f}s)")
                
            except Exception as e:
                # Use error recovery system
                error_result = self.error_recovery.handle_error(e, "document_processing")
                msgs.append(f"Error processing {os.path.basename(path)}: {error_result['message']}")
                if 'suggestion' in error_result:
                    msgs.append(f"Suggestion: {error_result['suggestion']}")
        
        if not all_chunks:
            msgs.append("No searchable content created from uploaded documents")
            return "\n".join(msgs)
        
        msgs.append(f"Building Advanced Search Index ({len(all_chunks)} chunks)")
        self._last_chunks = all_chunks
        self.vs.build(all_chunks)
        self.total_chunks = len(all_chunks)
        self.document_stats = stats
        # capture components for the table
        self._set_components_table()
        msgs.append("System Ready")
        return "\n".join(msgs)

    def get_document_sections(self):
        """Get available document sections for UI display"""
        if not self.document_sections:
            return "No documents processed yet."
        
        summary = "**Available Document Sections**\n\n"
        for filename, sections in self.document_sections.items():
            summary += f"**File: {filename}**\n"
            for section_name, section_info in sections.items():
                pages = section_info["pages"]
                page_range = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
                summary += f"  {section_name.replace('_', ' ').title()} ({page_range})\n"
            summary += "\n"
        
        return summary

    def answer(self, question, history, k=5, confidence_threshold=0.3, use_expansion=True, max_answer_tokens=0, enable_reranking=True, show_debug=True):
        # Primary path: LangGraph orchestration (with LangChain prompts)
        if getattr(self, "langgraph_orchestrator", None) and self.langgraph_orchestrator.available:
            try:
                return self.langgraph_orchestrator.answer(
                    question=question,
                    history=history,
                    k=k,
                    confidence_threshold=confidence_threshold,
                    use_expansion=use_expansion,
                    max_answer_tokens=max_answer_tokens,
                    enable_reranking=enable_reranking,
                    show_debug=show_debug,
                )
            except Exception as e:
                print(f"LangGraph pipeline failed, falling back to legacy flow: {e}")

        # Fallback path: existing legacy implementation
        if not question.strip():
            return "", history
        if not self.vs.chunks:
            return "", history + [(question, "Please upload and process documents first (left panel).")]

        # Start timing the entire response
        response_start = time.time()

        try:
            hits, retrieval_time = self.vs.search(
                question,
                top_k=int(k),
                use_expansion=bool(use_expansion),
                rerank=bool(enable_reranking)
            )

            if not hits:
                total_time = time.time() - response_start
                self.metrics.log_query(total_time, retrieval_time, 0.0, 0.0, 0.0, 0, "no_results", 0.0, False)
                return "", history + [(question, "No relevant information found in the uploaded documents.")]

            # Use higher confidence threshold for better quality
            threshold = max(0.3, float(confidence_threshold))  # Minimum 0.3
            max_relevance = max(hit.get("relevance", 0.0) for hit in hits)

            # Detect summary/general questions early
            question_lower = question.lower()
            is_summary_question = any(phrase in question_lower for phrase in [
                "what does the document say", "what is this about", "summarize", "what's this about", 
                "what is the document about", "what does it say", "summarize this", "give me a summary",
                "what is this document", "tell me about this document", "describe this document"
            ])

            final = None
            method = "extractive_qa"
            conf = 0.0
            qa_time = 0.0
            generation_time = 0.0

            # For summary questions, prioritize summary generation
            if is_summary_question and hits and max_relevance > 0.15:
                try:
                    summary, sum_time = self.gen.generate_summary(hits)
                    if summary and len(summary.split()) >= 5:
                        final = self._clean_answer(summary)
                        method = "generative_summary"
                        conf = 0.75
                        generation_time = sum_time
                except Exception as sum_e:
                    print(f"Summary generation error: {sum_e}")

            # If summary didn't work or not a summary question, try normal flow
            if not final:
                # 1) Try extractive QA first (precise answers)
                qa = self.qa.answer(question, hits)
                conf = qa.get("confidence", 0.0)
                qa_time = qa.get("processing_time", 0.0)

                if qa["found"] and conf >= threshold:
                    final = qa["answer"].strip()
                    final = self._clean_answer(final)
                else:
                    # 2) Fall back to Gemini (generative)
                    budget = int(max_answer_tokens) if max_answer_tokens and int(max_answer_tokens) > 0 else None
                    gen, gen_time = self.gen.generate(question, hits, max_tokens=budget)
                    generation_time = gen_time
                    if gen and len(gen.split()) >= 3 and not gen.lower().startswith(("insufficient", "error", "not stated")):
                        final = self._clean_answer(gen.strip())
                        method = "generative"
                        conf = 0.75 if self.gen.api_key_set else 0.3
                    else:
                        # 3) Extract relevant sentences
                        final = self._extract_relevant_sentences(question, hits)
                        method = "sentence_extraction"
                        conf = 0.5 if final != "Information not found in the provided documents." else 0.2

            # Final fallback: if still no good answer but have chunks, try summary
            if (not final or final.lower().startswith("information not found") or 
                final.lower().startswith("the requested information is not available")) and hits and max_relevance > 0.15:
                try:
                    summary, sum_time = self.gen.generate_summary(hits)
                    if summary and len(summary.split()) >= 5:
                        final = self._clean_answer(summary)
                        method = "generative_summary"
                        conf = 0.7
                        generation_time = generation_time + sum_time
                except Exception as sum_e:
                    print(f"Summary fallback error: {sum_e}")

            if not final or final.lower().startswith("information not found"):
                final = "The requested information is not available in the uploaded documents."

            # Calculate total response time
            total_time = time.time() - response_start

            # Log metrics
            has_relevant = len(hits) > 0 and max_relevance > 0.3
            self.metrics.log_query(
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                extraction_time=qa_time,
                max_relevance=max_relevance,
                answer_length=len(final),
                method=method,
                confidence=conf,
                has_relevant_results=has_relevant
            )

            # Format response with debug info if enabled
            if show_debug:
                debug_section = self.debug_info.format_debug_info(
                    hits, method, conf, total_time, retrieval_time, generation_time, question
                )
                final_response = final + debug_section
            else:
                final_response = final

            return "", history + [(question, final_response)]

        except Exception as e:
            total_time = time.time() - response_start
            self.metrics.log_query(total_time, 0.0, 0.0, 0.0, 0.0, 0, "error", 0.0, False)
            return "", history + [(question, f"Error processing question: {e}")]

    def _clean_answer(self, text):
        """Comprehensive answer cleaning"""
        if not text:
            return ""

        # Remove all types of citations and references
        text = re.sub(r'\[[^\]]*\]', '', text)  # [anything]
        text = re.sub(r'\([^)]*\.pdf[^)]*\)', '', text, flags=re.I)  # (anything.pdf)
        text = re.sub(r'[A-Za-z\s]*\.pdf[#\w]*', '', text, flags=re.I)  # filename.pdf references
        text = re.sub(r'#[a-f0-9]+', '', text)  # hash references
        text = re.sub(r'\(relevance\s*[0-9.]+\)', '', text)  # relevance scores
        text = re.sub(r'Method:\s*\w+\s*\|\s*Confidence:\s*[0-9.]+\s*\|\s*[0-9:]+', '', text)  # method info

        # Preserve readable structure instead of flattening everything into one line.
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Convert inline list markers to bullet lines.
        text = re.sub(r'(?m)^\s*[\*\u2022]\s+', '- ', text)      # line-start bullets
        text = re.sub(r'\s+\*\s+', '\n- ', text)                 # inline " * item"
        # Add paragraph breaks before inline numbered sections.
        text = re.sub(r'\s+([0-9]+\.)\s+', r'\n\n\1 ', text)
        # Normalize spaces per line.
        lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.split('\n')]
        cleaned_lines = []
        last_blank = False
        for ln in lines:
            if not ln:
                if not last_blank:
                    cleaned_lines.append("")
                last_blank = True
                continue
            cleaned_lines.append(ln)
            last_blank = False
        text = "\n".join(cleaned_lines).strip()

        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')) and not text.endswith(':'):
            text += "."

        return text

    def _extract_relevant_sentences(self, question, hits):
        """Extract relevant sentences as fallback"""
        q = (question or "").lower()
        q_terms = re.findall(r"[a-z0-9]+", q)
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "what", "where", "when", "which", "who", "how", "does", "did", "is",
            "are", "was", "were", "his", "her", "their", "into", "from"
        }
        q_terms = [w for w in q_terms if len(w) > 2 and w not in stop]
        if not q_terms:
            return "Information not found in the provided documents."

        # Generic phrase hints for common factoid intents.
        intent_phrases = []
        if "where" in q:
            intent_phrases.extend(["at", "in", "location", "institution", "place"])
        if "when" in q:
            intent_phrases.extend(["year", "date", "time", "period"])
        if "who" in q:
            intent_phrases.extend(["name", "person", "individual", "author"])
        if any(t in q for t in ("program", "course", "track", "major")):
            intent_phrases.extend(["program", "course", "track", "major"])

        best = None
        best_score = 0.0
        for h in hits[:6]:
            sentences = safe_sent_tokenize(h.get("text", ""))
            for sent in sentences:
                if len(sent.split()) < 5:
                    continue
                sent_lower = sent.lower()
                sent_terms = set(re.findall(r"[a-z0-9]+", sent_lower))
                overlap = sum(1 for t in q_terms if t in sent_terms)
                phrase_boost = sum(1 for p in intent_phrases if p in sent_lower)
                score = float(overlap) + (0.35 * float(phrase_boost))
                # Prefer concrete answer-like sentences over generic labels.
                if re.fullmatch(r"[a-z0-9 \-:/]{1,40}", sent_lower.strip()):
                    score -= 0.4
                if score > best_score:
                    best_score = score
                    best = sent.strip()

        if not best or best_score < 1.0:
            return "Information not found in the provided documents."

        best_sentence = self._clean_answer(best)
        return best_sentence if best_sentence else "Information not found in the provided documents."

    def explain_document(self):
        """Provide a summary of processed documents"""
        if not self.processed:
            return "No documents have been processed yet. Upload PDF files first."
        
        lines = ["Document Summary", ""]
        pages = sum(d['pages'] for d in self.processed)
        chunks = sum(d['chunks'] for d in self.processed)
        lines.append(f"**Overview:** Processed {len(self.processed)} document(s), {pages} page(s), split into {chunks} chunks.")
        lines.append("")
        
        lines.append("**Documents Processed:**")
        for d in self.processed:
            dist = ", ".join([f"{cnt} {k}" for k,cnt in d['chunk_distribution'].items()])
            lines.append(f"- {d['file']}: {d['chunks']} chunks ({dist}) from {d['pages']} pages")
            lines.append(f"  Extraction method: {d.get('extraction_method', 'unknown')}")
        
        if self.document_sections:
            lines.append("")
            lines.append("**Document Sections Detected:**")
            for filename, sections in self.document_sections.items():
                lines.append(f"- {filename}: {len(sections)} sections")
                for section_name in sections.keys():
                    lines.append(f"  • {section_name.replace('_', ' ').title()}")
        
        lines.append("")
        lines.append("**Next Steps:** Ask questions about your documents to get specific information.")
        
        return "\n".join(lines)

    def stats(self):
        if not self.processed:
            return "No documents processed yet."

        # Build components table (plain text, clean)
        components = self.components_table or {}
        table_lines = []
        if components:
            table_lines.append("Components Table")
            widest = max(len(k) for k in components.keys())
            for k,v in components.items():
                table_lines.append(f"- {k:{widest}} : {v}")
            table_lines.append("")

        s = ["Enhanced RAG System Information","",
           f"Documents: {len(self.processed)}", f"Total Chunks: {self.total_chunks}",
           f"Total Pages: {sum(doc['pages'] for doc in self.processed)}",
           f"Text Processed: {self.document_stats.get('total_text_length',0):,} characters",""]
        s.extend(table_lines)

        # Add performance metrics
        if self.metrics.queries_processed > 0:
            s.append("")
            s.extend(self.metrics.get_stats().split('\n'))
            s.append("")

        s.append("Documents:")
        for d in self.processed:
            dist = ", ".join([f"{cnt} {k}" for k,cnt in d['chunk_distribution'].items()])
            ext_time = d.get('extraction_time', 0.0)
            s.append(f"- {d['file']} — {d['chunks']} chunks ({dist}) from {d['pages']} pages [{d.get('extraction_method','unknown')}] ({ext_time:.2f}s)")
        
        # Add section information
        if self.document_sections:
            s.append("")
            s.append("Document Sections:")
            for filename, sections in self.document_sections.items():
                s.append(f"- {filename}: {len(sections)} sections")
                for section_name in sections.keys():
                    s.append(f"  • {section_name.replace('_', ' ').title()}")
        
        return "\n".join(s)

    def get_performance_metrics(self):
        """Return just the performance metrics"""
        return self.metrics.get_stats()

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset_metrics()
        return "Performance metrics reset."

    # Speech and Language Processing Methods
    def speech_to_text(self):
        """Convert speech to text using microphone"""
        if not self.speech_processor.available:
            return "Speech recognition not available. Please install required packages."
        return self.speech_processor.speech_to_text()
    
    def text_to_speech(self, text, language='en'):
        """Convert text to speech and play it"""
        if not self.speech_processor.available:
            return "Text-to-speech not available. Please install required packages."
        return self.speech_processor.text_to_speech(text, language)
    
    def detect_language(self, text):
        """Detect the language of input text"""
        if not self.bilingual_processor.available:
            return "Language detection not available. Please install required packages."
        lang_code = self.bilingual_processor.detect_language(text)
        lang_name = self.bilingual_processor.get_language_name(lang_code)
        return f"Detected language: {lang_name} ({lang_code})"
    
    def translate_text(self, text, target_lang='en', source_lang='auto'):
        """Translate text to target language"""
        if not self.bilingual_processor.available:
            return "Translation not available. Please install required packages."
        translated = self.bilingual_processor.translate_text(text, target_lang, source_lang)
        target_name = self.bilingual_processor.get_language_name(target_lang)
        return f"Translated to {target_name}: {translated}"
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        if not self.bilingual_processor.available:
            return "Language support not available. Please install required packages."
        languages = []
        for code, name in self.bilingual_processor.supported_languages.items():
            languages.append(f"{code}: {name}")
        return "Supported Languages:\n" + "\n".join(languages)
    
    def answer_with_translation(self, question, history, target_lang='en', k=5, confidence_threshold=0.3, use_expansion=True, max_answer_tokens=0, enable_reranking=True, show_debug=True):
        """Answer question and provide translation"""
        # First, detect the language of the question
        if self.bilingual_processor.available:
            source_lang = self.bilingual_processor.detect_language(question)
            if source_lang != 'en':
                # Translate question to English for processing
                question_en = self.bilingual_processor.translate_text(question, 'en', source_lang)
                print(f"Translated question: {question_en}")
            else:
                question_en = question
        else:
            question_en = question
        
        # Get answer in English
        if target_lang == 'en':
            # Use regular answer method
            return self.answer(question_en, history, k, confidence_threshold, use_expansion, max_answer_tokens, enable_reranking, show_debug)
        else:
            # Get answer and then translate
            empty, new_history = self.answer(question_en, history, k, confidence_threshold, use_expansion, max_answer_tokens, enable_reranking, show_debug)
            
            if new_history and len(new_history) > 0:
                last_question, last_answer = new_history[-1]
                
                # Translate the answer to target language
                if self.bilingual_processor.available:
                    translated_answer = self.bilingual_processor.translate_text(last_answer, target_lang, 'en')
                    target_name = self.bilingual_processor.get_language_name(target_lang)
                    
                    # Create bilingual response
                    bilingual_response = f"[{target_name} Translation]\n{translated_answer}\n\n[Original English]\n{last_answer}"
                    
                    # Update the last answer with bilingual response
                    new_history[-1] = (last_question, bilingual_response)
            
            return "", new_history

    def clear(self):
        self.__init__()
        return ("System cleared. Upload PDFs to start again.", [], "Ready.")

    # Export Methods
    def export_conversation(self, conversation_history, format_type="pdf", filename=None):
        """Export conversation to specified format"""
        return self.export_manager.export_conversation(conversation_history, format_type, filename)

    def get_export_formats(self):
        """Get available export formats"""
        return self.export_manager.get_export_formats()

    # Error Recovery Methods
    def handle_error_with_recovery(self, error, context="", retry_count=0):
        """Handle errors with recovery strategies"""
        return self.error_recovery.handle_error(error, context, retry_count)
    
    def get_error_statistics(self):
        """Get error statistics"""
        return self.error_recovery.get_error_stats()
    
    def reset_error_counts(self):
        """Reset error counts"""
        self.error_recovery.reset_error_counts()

    # ========== NEW FEATURES: Document Preview, Quick Templates, Answer Refinement, Comparison ==========
    
    def get_document_preview(self, filename=None):
        """Get full extracted text of a document for preview"""
        if not self.processed:
            return "No documents processed yet."
        if filename:
            if filename in self.raw_texts:
                txt = self._format_preview_text(self.raw_texts[filename])
                return f"## {filename}\n\n{txt}" if txt else f"## {filename}\n\n_No text found._"
            return f"Document '{filename}' not found."
        # Return all documents
        preview = []
        for doc in self.processed:
            fname = doc['file']
            if fname in self.raw_texts:
                txt = self._format_preview_text(self.raw_texts[fname])
                preview.append(f"## {fname}\n\n{txt}")
        return "\n\n".join(preview) if preview else "No document text available."

    def _format_preview_text(self, text):
        """Make extracted text easier to read in preview."""
        t = (text or "").strip()
        if not t:
            return ""
        # Normalize whitespace but preserve rough structure markers.
        t = t.replace("\r\n", "\n")
        t = re.sub(r"[ \t]+", " ", t)
        # Improve bullet readability.
        t = t.replace("•", "\n- ")
        # Page separators on separate lines.
        t = re.sub(r"(---\s*Page\s+\d+.*?---)", r"\n\n\1\n", t)
        # Add spacing before common section-like all-caps headers.
        t = re.sub(r"\n?([A-Z][A-Z &/]{3,})\s", r"\n\n**\1** ", t)
        # Collapse excessive blank lines.
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def browse_chunks(self, filename=None, chunk_type=None, limit=20):
        """Browse document chunks with metadata"""
        if not self.vs.chunks:
            return "No chunks available. Process documents first."
        chunks_info = []
        for i, chunk in enumerate(self.vs.chunks):
            meta = chunk.get("metadata", {})
            source_file = meta.get("source_file", "unknown")
            if filename and source_file != filename:
                continue
            if chunk_type and meta.get("chunk_type") != chunk_type:
                continue
            chunk_text = chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            chunks_info.append(
                f"### Chunk {i}\n"
                f"- Source: `{source_file}`\n"
                f"- Type: `{meta.get('chunk_type', 'unknown')}`\n\n"
                f"```\n{chunk_text}\n```"
            )
            if len(chunks_info) >= limit:
                break
        return "\n\n---\n\n".join(chunks_info) if chunks_info else "No matching chunks found."

    def get_document_segmentation(self, filename=None):
        """Show how documents were segmented into chunks"""
        if not self.processed:
            return "No documents processed yet."
        seg_info = []
        for doc in self.processed:
            fname = doc['file']
            if filename and fname != filename:
                continue
            dist = doc.get('chunk_distribution', {})
            seg_info.append(f"## {fname}")
            seg_info.append(f"- Total chunks: **{doc['chunks']}**")
            seg_info.append(f"- Distribution: {', '.join([f'{cnt} {k}' for k, cnt in dist.items()])}")
            seg_info.append(f"- Extraction: `{doc.get('extraction_method', 'unknown')}`")
            if fname in self.document_sections:
                seg_info.append(f"- Sections: **{len(self.document_sections[fname])}**")
                for sec_name in self.document_sections[fname].keys():
                    seg_info.append(f"  - {sec_name.replace('_', ' ').title()}")
            seg_info.append("")
        return "\n".join(seg_info)

    def quick_summarize(self, history):
        """Quick template: Summarize document"""
        empty, new_history = self.answer("Summarize this document. What are the main points?", history, k=8, confidence_threshold=0.2)
        return "", new_history

    def quick_key_points(self, history):
        """Quick template: Key points"""
        empty, new_history = self.answer("What are the key points and main highlights in this document?", history, k=8, confidence_threshold=0.2)
        return "", new_history

    def quick_main_topics(self, history):
        """Quick template: Main topics"""
        empty, new_history = self.answer("What are the main topics and subjects covered in this document?", history, k=8, confidence_threshold=0.2)
        return "", new_history

    def refine_answer(self, question, history, context_hits=None):
        """Refine/expand an answer with more detail"""
        if not history or len(history) == 0:
            empty, new_history = ("", history + [("", "No previous answer to refine. Ask a question first.")])
            return empty, new_history
        last_q, last_a = history[-1]
        refined_q = f"Based on the previous answer: '{last_a[:100]}...', provide more detailed information about: {question}. Include specific examples and context."
        empty, new_history = self.answer(refined_q, history[:-1], k=10, confidence_threshold=0.2)
        return empty, new_history

    def get_followup_suggestions(self, question, hits):
        """Generate follow-up question suggestions based on context"""
        if not hits:
            return []
        # Extract key topics from hits
        topics = set()
        for h in hits[:5]:
            text = h.get("text", "").lower()
            # Simple keyword extraction (could be improved)
            words = [w for w in text.split() if len(w) > 4 and w.isalpha()]
            topics.update(words[:5])
        suggestions = [
            f"Tell me more about {list(topics)[0] if topics else 'this topic'}",
            "What are the key details?",
            "Can you provide examples?",
            "What else should I know?",
        ]
        return suggestions[:4]

    def compare_documents(self, doc1_name, doc2_name, question=None):
        """Compare information across two documents"""
        if not self.processed or len(self.processed) < 2:
            return "Need at least 2 documents to compare."
        doc1_text = self.raw_texts.get(doc1_name, "")
        doc2_text = self.raw_texts.get(doc2_name, "")
        if not doc1_text or not doc2_text:
            return f"One or both documents not found: {doc1_name}, {doc2_name}"
        # Create comparison prompt
        doc1_ctx = self._safe_truncate_for_prompt(doc1_text, 5000)
        doc2_ctx = self._safe_truncate_for_prompt(doc2_text, 5000)

        if question:
            comp_q = (
                f"Compare the following question across these two documents:\n\n"
                f"Question: {question}\n\n"
                f"Document 1 ({doc1_name}):\n{doc1_ctx}\n\n"
                f"Document 2 ({doc2_name}):\n{doc2_ctx}\n\n"
                "Return STRICTLY markdown with this exact structure:\n"
                "## 1) Similarities\n"
                "- bullet points only\n\n"
                "## 2) Key Differences\n"
                "- bullet points only\n\n"
                "## 3) Unique Information in Each\n"
                "### Unique to Document 1\n"
                "- bullet points only\n"
                "### Unique to Document 2\n"
                "- bullet points only\n\n"
                "Keep bullets concise and readable."
            )
        else:
            comp_q = (
                "Compare these two documents and highlight similarities, differences, and unique points.\n\n"
                f"Document 1 ({doc1_name}):\n{doc1_ctx}\n\n"
                f"Document 2 ({doc2_name}):\n{doc2_ctx}\n\n"
                "Return STRICTLY markdown with this exact structure:\n"
                "## 1) Similarities\n"
                "- bullet points only\n\n"
                "## 2) Key Differences\n"
                "- bullet points only\n\n"
                "## 3) Unique Information in Each\n"
                "### Unique to Document 1\n"
                "- bullet points only\n"
                "### Unique to Document 2\n"
                "- bullet points only\n\n"
                "Keep bullets concise and readable."
            )
        # Use Gemini to generate comparison
        if self.gen.api_key_set and self.gen.model:
            try:
                hits = [{"text": self._safe_truncate_for_prompt(doc1_text, 2500), "metadata": {"source_file": doc1_name}},
                       {"text": self._safe_truncate_for_prompt(doc2_text, 2500), "metadata": {"source_file": doc2_name}}]
                comparison, _ = self.gen.generate(comp_q, hits)
                if not comparison:
                    return "Could not generate comparison."
                return self._format_comparison_markdown(comparison, doc1_name, doc2_name)
            except Exception as e:
                return f"Comparison error: {e}"
        return "Gemini not configured. Set API key to use document comparison."

    def _safe_truncate_for_prompt(self, text, max_chars=4000):
        """Truncate text near a word boundary to avoid broken fragments."""
        if not text or len(text) <= max_chars:
            return text or ""
        cut = text[:max_chars]
        last_ws = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
        if last_ws > int(max_chars * 0.7):
            cut = cut[:last_ws]
        return cut.rstrip() + " ..."

    def _format_comparison_markdown(self, text, doc1_name, doc2_name):
        """Format comparison response into readable markdown."""
        t = (text or "").strip()
        if not t:
            return "Could not generate comparison."

        # Preserve existing line breaks and normalize inconsistent section markup.
        t = t.replace("\r\n", "\n")
        t = re.sub(r"^\s*Here is a comparison of the two documents:\s*", "", t, flags=re.I)

        # Insert section breaks even when model returns inline numbering.
        t = re.sub(r"(?i)\s*(?:\*\*)?\s*1[\)\.]?\s*similarities\s*(?:\*\*)?\s*", "\n\n## 1) Similarities\n", t)
        t = re.sub(r"(?i)\s*(?:\*\*)?\s*2[\)\.]?\s*key differences\s*(?:\*\*)?\s*", "\n\n## 2) Key Differences\n", t)
        t = re.sub(r"(?i)\s*(?:\*\*)?\s*3[\)\.]?\s*unique information in each\s*(?:\*\*)?\s*", "\n\n## 3) Unique Information in Each\n", t)

        # Normalize Unique-to subsections.
        t = re.sub(r"(?i)\s*unique to document 1\s*(\([^)]*\))?\s*:?\s*", r"\n\n### Unique to Document 1 \1\n", t)
        t = re.sub(r"(?i)\s*unique to document 2\s*(\([^)]*\))?\s*:?\s*", r"\n\n### Unique to Document 2 \1\n", t)

        # Normalize bullet markers.
        t = re.sub(r"(?m)^\s*[\*\u2022]\s+", "- ", t)
        # Convert inline "* item * item" to bullets.
        t = re.sub(r"\s+\*\s+", "\n- ", t)
        # Remove stray markdown fence markers that sometimes appear inline.
        t = re.sub(r"(?m)^\s*#{1,6}\s*$", "", t)
        t = t.replace("##\n", "\n").replace("###\n", "\n")
        t = re.sub(r"\s+##\s*", "\n\n", t)
        t = re.sub(r"\s+###\s*", "\n\n", t)

        # If large text blocks remain under section headings, split into sentence bullets for readability.
        def _bullets_after_heading(md: str, heading: str) -> str:
            parts = md.split(heading)
            if len(parts) < 2:
                return md
            pre = parts[0]
            post = heading + parts[1]
            # Only process immediate block under heading if no bullets yet.
            block_match = re.search(rf"({re.escape(heading)}\n)(.+?)(\n## |\n### |\Z)", post, flags=re.S)
            if not block_match:
                return md
            head, body, tail_marker = block_match.group(1), block_match.group(2).strip(), block_match.group(3)
            if re.search(r"(?m)^\s*-\s+", body):
                return md
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 8]
            if not sentences:
                return md
            bullets = "\n".join(f"- {s}" for s in sentences[:8])
            rebuilt = head + bullets + "\n" + tail_marker
            return post[:block_match.start()] + rebuilt + post[block_match.end():]

        t = _bullets_after_heading(t, "## 1) Similarities")
        t = _bullets_after_heading(t, "## 2) Key Differences")
        t = _bullets_after_heading(t, "### Unique to Document 1")
        t = _bullets_after_heading(t, "### Unique to Document 2")

        # Cleanup spacing and duplicate headings.
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        header = f"## Comparison\n\n- Document 1: `{doc1_name}`\n- Document 2: `{doc2_name}`\n"
        return f"{header}\n\n{t}"
