"""Lectura de .docx/.pdf/.md/.txt y troceado en fragmentos con ubicación (página o encabezado)."""
from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_CHARS, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    loc: str   # "archivo · p. 12" o "archivo · Encabezado"


def _paragraphs_docx(path: Path):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    heading = ""
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            if (p.style.name or "").lower().startswith(("heading", "título", "titulo")):
                heading = text
            yield text, heading
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    yield " | ".join(cells), heading


def _paragraphs_pdf(path: Path):
    from pypdf import PdfReader

    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        text = page.extract_text() or ""
        for para in text.split("\n\n") if "\n\n" in text else [text]:
            para = " ".join(para.split())
            if para:
                yield para, f"p. {i}"


def _paragraphs_md(path: Path):
    import re

    heading, in_code = "", False
    para: list[str] = []

    def emit():
        text = " ".join(" ".join(para).split())
        para.clear()
        return text

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            if para and (t := emit()):
                yield t, heading
            heading = m.group(1)
            continue
        if not line.strip():
            if para and (t := emit()):
                yield t, heading
            continue
        para.append(line.strip())
    if para and (t := emit()):
        yield t, heading


def _split_long(text: str):
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        yield text[i:i + CHUNK_CHARS]


def _paragraphs(path: Path):
    ext = path.suffix.lower()
    if ext == ".docx":
        return _paragraphs_docx(path)
    if ext == ".pdf":
        return _paragraphs_pdf(path)
    if ext in (".md", ".markdown", ".txt"):
        return _paragraphs_md(path)
    raise ValueError(f"Formato no soportado: {ext} (usa .docx, .pdf o .md)")


def document_text(path: Path) -> str:
    """Texto completo con marcas de página o sección, para dárselo entero a Claude."""
    lines, last = [], None
    for text, loc in _paragraphs(path):
        if loc and loc != last:
            lines.append(f"\n[{loc}]" if path.suffix.lower() == ".pdf" else f"\n## {loc}")
            last = loc
        lines.append(text)
    return "\n".join(lines).strip()


def load_document(path: Path) -> tuple[list[Chunk], list[str]]:
    """Devuelve (fragmentos, esquema de encabezados)."""
    ext = path.suffix.lower()
    paras = _paragraphs(path)

    chunks: list[Chunk] = []
    outline: list[str] = []
    buf, buf_loc = "", ""

    def flush():
        nonlocal buf, buf_loc
        if buf.strip():
            chunks.append(Chunk(buf.strip(), f"{path.name} · {buf_loc}"))
        buf, buf_loc = "", ""

    for text, loc in paras:
        if ext != ".pdf" and loc and (not outline or outline[-1] != loc):
            outline.append(loc)
            flush()   # cada sección empieza un fragmento nuevo
        for piece in _split_long(text) if len(text) > CHUNK_CHARS else [text]:
            if buf and len(buf) + len(piece) > CHUNK_CHARS:
                tail = buf[-CHUNK_OVERLAP:]
                flush()
                buf, buf_loc = tail + "\n", loc
            if not buf_loc:
                buf_loc = loc
            buf += piece + "\n"
    flush()
    return chunks, outline
