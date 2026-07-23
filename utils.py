# Utilities for Enhanced RAG Chatbot
import os, re, hashlib, time, json, math, warnings
warnings.filterwarnings("ignore")
import sys, subprocess, numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import uuid

# NLTK setup for sentence tokenization
import nltk
for pkg in ("punkt","punkt_tab","stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except: pass
from nltk.tokenize import sent_tokenize as _sent_tokenize
from nltk.corpus import stopwords as _stopwords

from config import DOMAIN_KEYWORDS, DOMAIN_NEGATIVE_KEYWORDS

def ends_sentence(text: str) -> bool:
    s = text.strip()
    return bool(s) and s[-1] in ".?!"

def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))

def safe_sent_tokenize(text):
    try: return _sent_tokenize(text)
    except Exception: return re.split(r'(?<=[.!?])\s+', text)

class DomainHeuristics:
    """Generic domain heuristics that work for all document types"""
    KEYWORDS = DOMAIN_KEYWORDS
    NEGATIVE = DOMAIN_NEGATIVE_KEYWORDS
    
    @staticmethod
    def domain_score(text: str) -> float:
        """Calculate domain relevance score - returns 0 by default for generic use"""
        if not DomainHeuristics.KEYWORDS and not DomainHeuristics.NEGATIVE:
            return 0.0  # No domain-specific scoring for generic documents
        t = text.lower()
        pos = sum(1 for k in DomainHeuristics.KEYWORDS if k in t) if DomainHeuristics.KEYWORDS else 0
        neg = sum(1 for k in DomainHeuristics.NEGATIVE if k in t) if DomainHeuristics.NEGATIVE else 0
        return 0.03 * pos - 0.05 * neg

class PerformanceMetrics:
    """Tracks and calculates various performance metrics for the RAG system"""

    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.queries_processed = 0
        self.response_times = []
        self.retrieval_times = []
        self.generation_times = []
        self.extraction_times = []
        self.relevance_scores = []
        self.answer_lengths = []
        self.methods_used = defaultdict(int)
        self.confidence_scores = []
        self.hit_rates = []  # Track if query found relevant results
        self.start_time = time.time()

    def log_query(self, response_time, retrieval_time, generation_time, extraction_time,
                  max_relevance, answer_length, method, confidence, has_relevant_results):
        """Log metrics for a single query"""
        self.queries_processed += 1
        self.response_times.append(response_time)
        self.retrieval_times.append(retrieval_time)
        self.generation_times.append(generation_time)
        self.extraction_times.append(extraction_time)
        self.relevance_scores.append(max_relevance)
        self.answer_lengths.append(answer_length)
        self.methods_used[method] += 1
        self.confidence_scores.append(confidence)
        self.hit_rates.append(1 if has_relevant_results else 0)

    def get_stats(self):
        """Calculate comprehensive performance statistics"""
        if self.queries_processed == 0:
            return "No queries processed yet."

        # Calculate averages and percentiles
        avg_response_time = np.mean(self.response_times)
        avg_retrieval_time = np.mean(self.retrieval_times)
        avg_generation_time = np.mean(self.generation_times)
        avg_extraction_time = np.mean(self.extraction_times)
        avg_relevance = np.mean(self.relevance_scores)
        avg_confidence = np.mean(self.confidence_scores)
        hit_rate = np.mean(self.hit_rates) * 100

        p95_response_time = np.percentile(self.response_times, 95)

        # Session duration
        session_duration = time.time() - self.start_time
        session_hours = session_duration / 3600

        stats = f"""
Performance Metrics | Session Duration: {session_hours:.1f}h
Results from {self.queries_processed} queries processed

**Retrieval Performance**
* **Hit Rate:** {hit_rate:.1f}% (percentage of queries with relevant results)
* **Average Relevance Score:** {avg_relevance:.3f}
* **Retrieval Latency:** {avg_retrieval_time*1000:.0f}ms average
* **P95 Response Time:** {p95_response_time:.2f}s

**End-to-End Accuracy**
* **Average Confidence:** {avg_confidence:.3f}
* **Method Distribution:** {dict(self.methods_used)}
* **Answer Completeness:** {np.mean(self.answer_lengths):.0f} chars average

**System Performance**
* **Average Response Time:** {avg_response_time:.2f} seconds
* **Document Processing:** {avg_extraction_time:.2f}s per query
* **Retrieval Latency:** {avg_retrieval_time*1000:.0f}ms
* **LLM Generation:** {avg_generation_time:.2f}s average
* **Queries per Hour:** {(self.queries_processed / max(session_hours, 0.01)):.1f}
"""
        return stats

class ResponseDebugInfo:
    """Formats debug information for responses"""

    @staticmethod
    def format_debug_info(hits, method, confidence, response_time, retrieval_time, generation_time, question=None):
        """Format detailed debug information about the response"""

        debug_info = "\n\n---\n"
        debug_info += f"**Method Used:** {method.replace('_', ' ').title()}\n"
        debug_info += f"**Confidence Score:** {confidence:.3f}\n"
        debug_info += f"**Response Time:** {response_time:.2f}s (Retrieval: {retrieval_time*1000:.0f}ms, Generation: {generation_time:.2f}s)\n\n"

        if hits:
            debug_info += "**Retrieved Chunks:**\n"
            q = (question or "").lower()
            q_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2]
            stop = {"the", "and", "for", "with", "from", "this", "that", "what", "where", "when", "which", "who", "how"}
            q_tokens = [t for t in q_tokens if t not in stop]

            # De-duplicate near-identical chunk texts (common with hierarchical chunk types).
            dedup_hits = []
            seen = set()
            for hit in hits:
                norm = re.sub(r"\s+", " ", hit.get("text", "")).strip().lower()
                # fingerprint first segment to collapse same-content different chunk-size views
                fp = norm[:320]
                if fp and fp not in seen:
                    seen.add(fp)
                    dedup_hits.append(hit)
                if len(dedup_hits) >= 3:
                    break

            for i, hit in enumerate(dedup_hits, 1):
                text = hit.get("text", "")
                preview = ""
                if text:
                    sentences = safe_sent_tokenize(text)
                    best_sent = ""
                    best_score = -1
                    for s in sentences:
                        s_clean = re.sub(r"\s+", " ", s).strip()
                        if len(s_clean.split()) < 5:
                            continue
                        sl = s_clean.lower()
                        score = sum(1 for t in q_tokens if t in sl)
                        if score > best_score:
                            best_score = score
                            best_sent = s_clean
                    # If no sentence overlap found, fallback to beginning preview.
                    if best_sent and best_score > 0:
                        preview = best_sent[:220]
                    else:
                        preview = re.sub(r"\s+", " ", text).strip()[:150]
                if len(preview) >= 150:
                    preview += "..."
                relevance = hit.get("relevance", 0.0)
                source_file = hit.get("source_file") or hit.get("metadata", {}).get("source_file", "unknown")
                chunk_type = hit.get("metadata", {}).get("chunk_type", "medium")
                debug_info += f'{i}. *"{preview}"*\n'
                debug_info += f'   Source: {source_file} | Type: {chunk_type} | Similarity: {relevance:.3f}\n'

        return debug_info

class ErrorRecovery:
    """Handles error recovery and graceful degradation"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.max_retries = 3
        self.recovery_strategies = {
            "api_error": self._handle_api_error,
            "processing_error": self._handle_processing_error,
            "memory_error": self._handle_memory_error,
            "timeout_error": self._handle_timeout_error
        }
    
    def handle_error(self, error, context="", retry_count=0):
        """Handle errors with recovery strategies"""
        error_type = self._classify_error(error)
        error_key = f"{context}_{error_type}"
        
        self.error_counts[error_key] += 1
        
        if retry_count < self.max_retries:
            recovery_result = self._attempt_recovery(error_type, error, context, retry_count)
            if recovery_result:
                return recovery_result
        
        # If recovery failed, return graceful degradation
        return self._graceful_degradation(error_type, context)
    
    def _classify_error(self, error):
        """Classify error type for appropriate recovery strategy"""
        error_str = str(error).lower()
        
        if any(word in error_str for word in ["api", "key", "authentication", "quota"]):
            return "api_error"
        elif any(word in error_str for word in ["memory", "out of memory", "insufficient"]):
            return "memory_error"
        elif any(word in error_str for word in ["timeout", "timed out", "too long"]):
            return "timeout_error"
        else:
            return "processing_error"
    
    def _attempt_recovery(self, error_type, error, context, retry_count):
        """Attempt to recover from error"""
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, context, retry_count)
        return None
    
    def _handle_api_error(self, error, context, retry_count):
        """Handle API-related errors"""
        if "quota" in str(error).lower():
            return {
                "recovered": False,
                "message": "API quota exceeded. Please try again later.",
                "suggestion": "Check your API usage limits or upgrade your plan."
            }
        elif "authentication" in str(error).lower():
            return {
                "recovered": False,
                "message": "Authentication failed. Please check your API key.",
                "suggestion": "Verify your API key is correct and has proper permissions."
            }
        else:
            return {
                "recovered": False,
                "message": "API service temporarily unavailable.",
                "suggestion": "Please try again in a few minutes."
            }
    
    def _handle_processing_error(self, error, context, retry_count):
        """Handle document processing errors"""
        if retry_count < 2:
            return {
                "recovered": True,
                "message": f"Retrying processing... (attempt {retry_count + 1})",
                "action": "retry"
            }
        else:
            return {
                "recovered": False,
                "message": "Document processing failed after multiple attempts.",
                "suggestion": "Try with a different document or check file format."
            }
    
    def _handle_memory_error(self, error, context, retry_count):
        """Handle memory-related errors"""
        return {
            "recovered": False,
            "message": "Insufficient memory to process this document.",
            "suggestion": "Try with a smaller document or close other applications."
        }
    
    def _handle_timeout_error(self, error, context, retry_count):
        """Handle timeout errors"""
        if retry_count < 2:
            return {
                "recovered": True,
                "message": f"Retrying with extended timeout... (attempt {retry_count + 1})",
                "action": "retry_extended"
            }
        else:
            return {
                "recovered": False,
                "message": "Processing timed out after multiple attempts.",
                "suggestion": "Document may be too large or complex to process."
            }
    
    def _graceful_degradation(self, error_type, context):
        """Provide graceful degradation when recovery fails"""
        degradation_messages = {
            "api_error": "Using offline processing mode. Some features may be limited.",
            "processing_error": "Switching to basic text extraction. Advanced features disabled.",
            "memory_error": "Processing with reduced quality to conserve memory.",
            "timeout_error": "Using quick processing mode. Results may be less comprehensive."
        }
        
        return {
            "recovered": False,
            "message": degradation_messages.get(error_type, "Service temporarily degraded."),
            "mode": "degraded"
        }
    
    def get_error_stats(self):
        """Get error statistics for monitoring"""
        return dict(self.error_counts)
    
    def reset_error_counts(self):
        """Reset error counts"""
        self.error_counts.clear()
