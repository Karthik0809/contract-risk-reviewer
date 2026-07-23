# LLM Interface module for Enhanced RAG Chatbot
import os
import re
import time
from datetime import datetime

from config import DEFAULT_GEMINI_MODEL, SUPPORTED_LANGUAGES

# Gemini via Google Generative AI only
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    print("Google Generative AI not available. Install: pip install google-generativeai")

# Speech and Language Processing
try:
    import speech_recognition as sr
    import pyttsx3
    from deep_translator import GoogleTranslator
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Speech features not available. Install: pip install SpeechRecognition pyttsx3 deep-translator")

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False
    print("Language detection not available. Install: pip install langdetect")

from utils import ends_sentence

class GeminiGenerator:
    def __init__(self):
        self.model_name = DEFAULT_GEMINI_MODEL
        self.api_key_set = False
        self.model = None
        self.api_key = None

    def _normalize_model_name(self, name: str) -> str:
        return (name or "").replace("models/", "").strip()

    def _list_generate_models(self):
        """Return model names that support generateContent for this key."""
        names = []
        try:
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    names.append(self._normalize_model_name(getattr(m, "name", "")))
        except Exception as e:
            print(f"Could not list Gemini models: {e}")
        # Deduplicate while preserving order
        dedup = []
        seen = set()
        for n in names:
            if n and n not in seen:
                dedup.append(n)
                seen.add(n)
        return dedup

    def _pick_best_model(self, preferred_name: str, available):
        pref = self._normalize_model_name(preferred_name)
        if pref in available:
            return pref

        # Prioritize stable flash/pro candidates if preferred is unavailable.
        priority_keywords = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "flash",
            "gemini-1.5-pro",
            "pro",
        ]
        for key in priority_keywords:
            for name in available:
                if key in name:
                    return name
        return available[0] if available else None

    def _rebind_model(self, candidate_name: str):
        if not candidate_name:
            return False
        try:
            self.model = genai.GenerativeModel(candidate_name)
            self.model_name = candidate_name
            return True
        except Exception as e:
            print(f"Failed to bind Gemini model '{candidate_name}': {e}")
            return False

    def _recover_model_if_not_found(self, error: Exception):
        """If the current model 404s, auto-pick another supported model and retry."""
        err = str(error).lower()
        if "not found" not in err and "404" not in err:
            return False
        available = self._list_generate_models()
        if not available:
            return False
        for name in available:
            if name != self.model_name and self._rebind_model(name):
                print(f"Switched Gemini model to available model: {self.model_name}")
                return True
        return False

    def configure(self, api_key: str):
        print(f"Configuring Gemini with model: '{self.model_name}'")
        if not api_key or not api_key.strip():
            print("No API key provided")
            return False
        api_key = api_key.strip()
        self.api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        if not GOOGLE_GENAI_AVAILABLE:
            print("google-generativeai not installed")
            return False
        try:
            genai.configure(api_key=api_key)
            available = self._list_generate_models()
            picked = self._pick_best_model(self.model_name, available)
            if not picked:
                print("No Gemini model with generateContent is available for this API key.")
                self.api_key_set = False
                return False
            self.model = genai.GenerativeModel(picked)
            self.model_name = picked
            self.api_key_set = True
            print(f"Gemini configured: {self.model_name}")
            return True
        except Exception as e:
            error_str = str(e)
            print(f"Gemini configure failed: {e}")
            if "API_KEY" in error_str or "401" in error_str or "Unauthorized" in error_str or "invalid" in error_str.lower():
                self.api_key_set = False
            return False

    def _prompt(self, question, hits):
        """Build concise, plain-English prompt with no inline citations"""
        primary = []
        for h in hits[:3]:
            txt = re.sub(r'\s+', ' ', h["text"]).strip()
            primary.append(txt[:2000])  # Increased context for more complete answers

        ctx = "\n\n".join(primary)
        sys = (
            "You are a helpful document assistant. "
            "Answer ONLY from the provided context. "
            "Write in clear, plain English without any references to files, chunks, or metadata. "
            "Provide complete and comprehensive answers that fully address the question. "
            "Do not cut off answers abruptly - ensure each answer is complete and well-formed. "
            "If the answer involves multiple parts, address each part thoroughly. "
            "If the answer is not present, say: 'Not stated in the provided documents.'"
        )
        usr = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer clearly and completely:"
        return f"{sys}\n\n{usr}"

    def _summary_prompt(self, hits):
        """Prompt for summarizing what the document(s) are about."""
        primary = []
        for h in hits[:8]:  # Use more chunks for better context
            txt = re.sub(r'\s+', ' ', h["text"]).strip()
            primary.append(txt[:2000])  # Longer chunks for comprehensive summary
        ctx = "\n\n".join(primary)
        return (
            "You are a helpful document assistant. Based ONLY on the following text excerpts from the user's uploaded document(s), "
            "write a comprehensive summary (3–7 sentences) of what the document(s) are about. "
            "Include: the main topic/subject, key roles or positions mentioned, important achievements or projects, technical skills or expertise, "
            "and any other significant details. Write in clear, complete sentences. "
            "Do NOT say 'information not available' or 'not stated' - the excerpts clearly contain content, so provide a meaningful summary.\n\n"
            f"Document Excerpts:\n{ctx}\n\n"
            "Summary (write a clear, comprehensive summary based on the excerpts above):"
        )

    def generate_summary(self, hits):
        """Generate a short summary of what the retrieved document excerpts are about."""
        gen_start = time.time()
        if not hits or not self.api_key_set or not self.model:
            return None, time.time() - gen_start
        try:
            prompt = self._summary_prompt(hits)
            response = self.model.generate_content(prompt)
            txt = (response.text or "").strip()
            txt = self._clean_response(txt)
            if txt and len(txt.split()) >= 5:
                return txt, time.time() - gen_start
        except Exception as e:
            if self._recover_model_if_not_found(e):
                try:
                    response = self.model.generate_content(prompt)
                    txt = (response.text or "").strip()
                    txt = self._clean_response(txt)
                    if txt and len(txt.split()) >= 5:
                        return txt, time.time() - gen_start
                except Exception:
                    pass
        return None, time.time() - gen_start

    def generate(self, question, hits, max_tokens=None):
        gen_start = time.time()
        if not hits:
            return "No relevant context available.", time.time() - gen_start

        if not self.api_key_set or not self.model:
            return "Generative answer unavailable: set a valid Google API key in the left panel.", time.time() - gen_start

        prompt = self._prompt(question, hits)

        try:
            response = self.model.generate_content(prompt)
            txt = (response.text or "").strip()

            # Aggressive cleaning of citations and metadata
            txt = self._clean_response(txt)

            if not txt or len(txt.split()) < 3:
                gen_time = time.time() - gen_start
                return "Insufficient content in the provided context.", gen_time

            gen_time = time.time() - gen_start
            return txt, gen_time
        except Exception as e:
            if self._recover_model_if_not_found(e):
                try:
                    response = self.model.generate_content(prompt)
                    txt = (response.text or "").strip()
                    txt = self._clean_response(txt)
                    if txt and len(txt.split()) >= 3:
                        return txt, time.time() - gen_start
                except Exception as inner_e:
                    e = inner_e
            gen_time = time.time() - gen_start
            return f"Error generating response: {e}", gen_time

    def _clean_response(self, text):
        """Aggressively clean response from any citations or metadata"""
        # Remove any bracketed content
        text = re.sub(r'\[[^\]]*\]', '', text)

        # Remove file references
        text = re.sub(r'[A-Za-z\s]*\.pdf[#\w]*', '', text, flags=re.I)

        # Remove chunk references
        text = re.sub(r'#[a-f0-9]+', '', text)

        # Remove relevance scores
        text = re.sub(r'\(relevance\s*[0-9.]+\)', '', text)

        # Remove method/confidence info
        text = re.sub(r'Method:\s*\w+\s*\|\s*Confidence:\s*[0-9.]+\s*\|\s*[0-9:]+', '', text)

        # Preserve line structure for readability in chat UI.
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        # Ensure proper sentence ending
        if text and not ends_sentence(text) and len(text.split()) > 4:
            text += "."

        return text

class SpeechProcessor:
    """Handles speech-to-text and text-to-speech functionality"""
    
    def __init__(self):
        if not SPEECH_AVAILABLE:
            self.available = False
            self.stt_available = False
            self.tts_available = False
            self.recognizer = None
            self.tts_engine = None
            return

        self.recognizer = None
        self.tts_engine = None
        self.stt_available = False
        self.tts_available = False

        # STT can be available even if TTS is unavailable in cloud containers.
        try:
            self.recognizer = sr.Recognizer()
            self.stt_available = True
        except Exception as e:
            print(f"Speech recognition initialization failed: {e}")

        # pyttsx3 may fail on Linux containers when eSpeak is missing.
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_available = True
        except Exception as e:
            self.tts_engine = None
            self.tts_available = False
            print(f"Text-to-speech disabled: {e}")

        self.available = self.stt_available or self.tts_available
        if not self.tts_available:
            return
        
        # Configure TTS engine
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to set a good default voice
            for voice in voices:
                if 'en' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 150)  # Words per minute
        self.tts_engine.setProperty('volume', 0.9)  # Volume level
    
    def speech_to_text(self, audio_data=None):
        """Convert speech to text using microphone or audio file"""
        if not self.stt_available or self.recognizer is None:
            return "Speech recognition not available. Please install required packages."
        
        try:
            if audio_data is None:
                # Use microphone
                with sr.Microphone() as source:
                    print("Listening... Speak now!")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            else:
                # Use provided audio data
                audio = audio_data
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            return text.strip()
            
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Could not understand the speech. Please try again."
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"
        except Exception as e:
            return f"Speech recognition error: {e}"
    
    def text_to_speech(self, text, language='en'):
        """Convert text to speech and play it"""
        if not self.tts_available or self.tts_engine is None:
            return "Text-to-speech not available in this environment."
        
        try:
            # Set language-specific voice if available
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a voice for the specified language
                for voice in voices:
                    if language in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Speak the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return f"Spoken: {text[:100]}{'...' if len(text) > 100 else ''}"
            
        except Exception as e:
            return f"Text-to-speech error: {e}"

class BilingualProcessor:
    """Handles bilingual Q&A support with translation"""
    
    def __init__(self):
        if not LANG_DETECT_AVAILABLE:
            self.available = False
            return
        
        self.available = True
        self.supported_languages = SUPPORTED_LANGUAGES
    
    def detect_language(self, text):
        """Detect the language of input text"""
        if not self.available:
            return 'en'  # Default to English
        
        try:
            lang = detect(text)
            return lang if lang in self.supported_languages else 'en'
        except LangDetectException:
            return 'en'
    
    def translate_text(self, text, target_lang='en', source_lang='auto'):
        """Translate text to target language"""
        if not self.available:
            return text  # Return original text if translation not available
        
        try:
            if source_lang == 'auto':
                source_lang = self.detect_language(text)
            
            if source_lang == target_lang:
                return text
            
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translation = translator.translate(text)
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def get_language_name(self, lang_code):
        """Get human-readable language name from code"""
        return self.supported_languages.get(lang_code, 'Unknown')
    
    def is_supported_language(self, lang_code):
        """Check if language is supported"""
        return lang_code in self.supported_languages
