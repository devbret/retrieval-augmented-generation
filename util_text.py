import os, re, math
from typing import List, Tuple
from pypdf import PdfReader
import markdown

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def ocr_pdf(path: str) -> str:
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError:
        return ""
    parts = []
    pdf = pdfium.PdfDocument(path)
    try:
        for page in pdf:
            bitmap = page.render(scale=300 / 72)
            try:
                parts.append(pytesseract.image_to_string(bitmap.to_pil()))
            except Exception:
                parts.append("")
    finally:
        pdf.close()
    return "\n".join(parts)

def extract_pdf_tables(path: str) -> List[str]:
    try:
        import pdfplumber
    except ImportError:
        return []
    blocks = []
    try:
        with pdfplumber.open(path) as pdf:
            for pageno, page in enumerate(pdf.pages, 1):
                for table in page.extract_tables() or []:
                    rows = []
                    for row in table:
                        cells = [(c or "").strip().replace("\n", " ") for c in row]
                        if any(cells):
                            rows.append(" | ".join(cells))
                    if len(rows) > 1:
                        blocks.append(f"[Table on page {pageno}]\n" + "\n".join(rows))
    except Exception:
        return blocks
    return blocks

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n".join(parts)

    if len(text.strip()) < 20:
        ocr_text = ocr_pdf(path)
        if ocr_text.strip():
            return ocr_text
        return text

    tables = extract_pdf_tables(path)
    if tables:
        text = text + "\n\n" + "\n\n".join(tables)
    return text

def load_md(path: str) -> str:
    raw = load_text(path)
    return raw

def load_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".md", ".markdown"):
        return load_md(path)
    return load_text(path)

def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current: List[str] = []

    def current_token_count() -> int:
        if not current:
            return 0
        return sum(len(p.split()) for p in current)

    def flush():
        if current:
            joined = " ".join(current).strip()
            if joined:
                chunks.append(joined)

    for para in paragraphs:
        tokens = para.split()
        token_len = len(tokens)

        if token_len > chunk_size:
            flush()
            i = 0
            step = max(1, chunk_size - overlap)
            while i < token_len:
                sub_chunk = " ".join(tokens[i:i+chunk_size])
                if sub_chunk.strip():
                    chunks.append(sub_chunk)
                i += step
            current = []
            continue

        if current_token_count() + token_len > chunk_size:
            flush()
            current = [para]
        else:
            current.append(para)

    flush()

    return chunks

def yield_files(root: str) -> List[str]:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.startswith("."): 
                continue
            path = os.path.join(dirpath, name)
            if os.path.getsize(path) == 0:
                continue
            yield path
