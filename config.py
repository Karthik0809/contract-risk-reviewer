# Configuration file for Enhanced RAG Chatbot
# Generic document Q&A system for all types of PDFs
import os

# --------------------------- Configuration ---------------------------
DEFAULT_GEMINI_MODEL = os.environ.get("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash")

# Gradio server port (override with env GRADIO_SERVER_PORT)
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

# Pre-defined API keys (used on startup; override via UI if needed)
DEFAULT_GEMINI_API_KEY = os.environ.get("DEFAULT_GEMINI_API_KEY", "")
DEFAULT_PINECONE_API_KEY = os.environ.get("DEFAULT_PINECONE_API_KEY", "")

# Pinecone index (used when API key is set)
# Your index: "ragquery" (llama-text-embed-v2 integrated model)
# Note: If "ragquery" uses integrated embeddings, we'll create a separate index for local embeddings (768 dims)
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ragquery")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
# OpenAI embeddings disabled - using sentence-transformers (768 dims) instead
USE_OPENAI_EMBEDDINGS = os.environ.get("USE_OPENAI_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIM = int(os.environ.get("OPENAI_EMBEDDING_DIM", "1024"))

# --------------------------- System dependencies ---------------------------
# Note: Install required packages manually:
# pip install gradio pypdf pdf2image pytesseract pillow nltk rank-bm25 faiss-cpu
# pip install sentence-transformers transformers accelerate scikit-learn
# pip install google-generativeai llama-index llama-index-llms-gemini
# pip install SpeechRecognition pyttsx3 deep-translator langdetect
# pip install pyngrok  # For public tunneling

# Domain heuristics (generic - can be customized per domain)
# These are optional keywords that can boost relevance for specific domains
DOMAIN_KEYWORDS = set()  # Empty by default - works for all document types

DOMAIN_NEGATIVE_KEYWORDS = set()  # Empty by default

# Performance settings
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_USE_EXPANSION = True
DEFAULT_ENABLE_RERANKING = True
DEFAULT_MAX_TOKENS = 0
DEFAULT_SHOW_DEBUG = True

# OCR settings
DEFAULT_DPI = 300
OCR_CONFIG = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}\"' -/$%&*+=<>@#^_|\\~` "

# Chunking settings
TARGET_TOKENS = 300
OVERLAP_SENTENCES = 2
BOUNDARY_DROP = 0.15
TOK_CHARS_RATIO = 0.25

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'te': 'Telugu'
}

# Export formats
SUPPORTED_EXPORT_FORMATS = ["txt", "pdf", "docx"]

# Error recovery settings
MAX_RETRIES = 3
