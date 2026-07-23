# Document processing - pypdf only (no OCR, no poppler)
# -*- coding: utf-8 -*-
import os
import re
import sys
from pypdf import PdfReader

# Ensure UTF-8 handling for text extraction
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class PDFSectionSegmenter:
    def __init__(self):
        self.section_keywords = {
            "introduction": ["introduction", "overview", "summary", "executive summary", "abstract"],
            "main_content": ["main", "content", "body", "details", "information"],
            "financial": ["financial", "cost", "price", "fee", "payment", "amount", "total", "balance"],
            "legal": ["terms", "conditions", "agreement", "contract", "legal", "disclosure", "notice"],
            "contact": ["contact", "address", "phone", "email", "location"]
        }

    def segment_document(self, text, filename):
        sections = {}
        pages = text.split("--- Page")
        for section_name, keywords in self.section_keywords.items():
            section_content = []
            section_pages = []
            for i, page in enumerate(pages):
                if i == 0:
                    continue
                page_lower = page.lower()
                if sum(1 for k in keywords if k in page_lower) >= 2:
                    section_content.append(page)
                    section_pages.append(i)
            if section_content:
                sections[section_name] = {"content": "\n".join(section_content), "pages": section_pages, "filename": filename, "keyword_matches": len(keywords)}
        if not sections and len(pages) > 1:
            sections["general"] = {"content": text, "pages": list(range(1, len(pages))), "filename": filename, "keyword_matches": 0}
        return sections

    def get_section_summary(self, sections):
        """Return a short summary of section names for display."""
        if not sections:
            return "No sections detected."
        return ", ".join(s.replace("_", " ").title() for s in sections.keys())


class EnhancedDocumentProcessor:
    """Extract text from PDFs using pypdf only. No OCR, no poppler."""

    def __init__(self):
        self.segmenter = PDFSectionSegmenter()

    def extract_text(self, path: str):
        try:
            text, pages, structure = self._extract_digital(path)
        except Exception as e:
            return (
                f"Could not extract text from PDF: {e}. Ensure the file is a valid text-based PDF (not a scanned image).",
                {"extraction_method": "failed", "path": path}
            )
        text_clean = (text or "").strip()
        if len(text_clean) < 10:
            return (
                "This PDF has no extractable text (it may be scanned or image-only). Use a PDF that contains selectable text.",
                {"extraction_method": "failed", "path": path}
            )
        sections = self.segmenter.segment_document(text, os.path.basename(path))
        return text, {
            "file": os.path.basename(path),
            "path": path,
            "total_pages": pages,
            "extraction_method": "pypdf",
            "structure": structure,
            "sections": sections
        }

    def _extract_digital(self, path: str):
        reader = PdfReader(path)
        parts = []
        structure = {"sections": [], "tables": [], "lists": []}
        for i, p in enumerate(reader.pages):
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            self._detect_structure(t, i + 1, structure)
            parts.append(f"\n--- Page {i + 1} ---\n{self._clean_text(t)}")
        text = "\n".join(parts)
        if not text.strip():
            raise ValueError("PDF has no extractable text.")
        return text, len(reader.pages), structure

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Ensure text is Unicode string (handle any encoding issues)
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text, flags=re.I)
        return text.strip()

    def _detect_structure(self, text, page, structure):
        for i, line in enumerate((text or "").split("\n")):
            s = line.strip()
            if not s:
                continue
            if len(s) < 120 and (s.isupper() or re.match(r"^\d+\.\s+[A-Z]", s)):
                structure["sections"].append({"title": s, "page": page, "line": i})
            if re.match(r"^\s*[•\-*]|^\s*\d+\.", s):
                structure["lists"].append({"content": s, "page": page, "line": i})
