"""Índice híbrido: embeddings (coseno) + solapamiento de palabras. Caché en disco por archivo."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import llm
from .config import CACHE_DIR, CHUNK_CHARS, CHUNK_OVERLAP, EMBED_MODEL, TOP_K
from .docs import Chunk, document_text, load_document

_WORD = re.compile(r"\w{3,}", re.UNICODE)


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


class Index:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vecs = np.zeros((0, 1), dtype=np.float32)
        self.tokens: list[set[str]] = []
        self.briefs: dict[str, str] = {}   # archivo -> resumen
        self.outlines: dict[str, list[str]] = {}
        self.texts: dict[str, str] = {}     # archivo -> texto completo
        self._vocab, self._vocab_n = "", -1

    def _cache_key(self, path: Path) -> Path:
        h = hashlib.sha1(path.read_bytes())
        h.update(f"{EMBED_MODEL}|{CHUNK_CHARS}|{CHUNK_OVERLAP}".encode())
        return CACHE_DIR / h.hexdigest()

    def add_document(self, path: Path, progress=lambda done, total: None):
        """Bloqueante (usar en hilo). Embebe y cachea; si ya estaba cacheado es instantáneo."""
        key = self._cache_key(path)
        meta, npy = key.with_suffix(".json"), key.with_suffix(".npy")
        if meta.exists() and npy.exists():
            data = json.loads(meta.read_text())
            chunks = [Chunk(**c) for c in data["chunks"]]
            outline, vecs = data["outline"], np.load(npy)
            progress(len(chunks), len(chunks))
        else:
            chunks, outline = load_document(path)
            if not chunks:
                raise ValueError("No se pudo extraer texto del documento (¿PDF escaneado?)")
            out = []
            for i in range(0, len(chunks), 32):
                out += llm.embed_sync([c.text for c in chunks[i:i + 32]])
                progress(min(i + 32, len(chunks)), len(chunks))
            vecs = np.array(out, dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            np.save(npy, vecs)
            meta.write_text(json.dumps({"chunks": [c.__dict__ for c in chunks], "outline": outline}))
        self.chunks += chunks
        self.tokens += [_tokens(c.text) for c in chunks]
        self.vecs = vecs if len(self.vecs) == 0 else np.vstack([self.vecs, vecs])
        self.outlines[path.name] = outline
        txt = key.with_suffix(".txt")
        if not txt.exists():
            txt.write_text(document_text(path))
        self.texts[path.name] = txt.read_text()

    def full_corpus(self) -> str:
        """Todos los documentos completos, en orden estable (para que la caché de Claude no se invalide)."""
        return "\n\n".join(f'<documento nombre="{n}">\n{self.texts[n]}\n</documento>' for n in sorted(self.texts))

    def sample_text(self, name: str, n: int = 8, size: int = 1200) -> str:
        mine = [c for c in self.chunks if c.loc.startswith(name)]
        if not mine:
            return ""
        idx = np.linspace(0, len(mine) - 1, min(n, len(mine))).astype(int)
        return "\n---\n".join(mine[i].text[:size] for i in idx)

    async def search(self, query: str, k: int = TOP_K) -> list[Chunk]:
        if not self.chunks:
            return []
        q = np.array((await llm.embed([query]))[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        sims = self.vecs @ q
        qt = _tokens(query)
        if qt:
            overlap = np.array([len(qt & t) / len(qt) for t in self.tokens], dtype=np.float32)
            sims = sims + 0.15 * overlap
        return [self.chunks[i] for i in np.argsort(-sims)[:k]]

    def vocab_hint(self, max_chars: int = 450) -> str:
        """Términos técnicos de los documentos (siglas, CamelCase, encabezados) para guiar a Whisper."""
        import re
        from collections import Counter

        if self._vocab_n == len(self.chunks):
            return self._vocab
        counts = Counter()
        for c in self.chunks:
            counts.update(re.findall(r"\b(?:[A-Z]{2,}[a-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b", c.text))
        terms = [t for t, _ in counts.most_common(40)]
        for name in self.outlines:
            terms += [h for h in self.outlines[name][:10] if len(h) < 40]
        hint = ""
        for t in terms:
            if len(hint) + len(t) + 2 > max_chars:
                break
            hint += t + ", "
        self._vocab, self._vocab_n = hint.rstrip(", "), len(self.chunks)
        return self._vocab

    def brief(self) -> str:
        parts = []
        for name in self.outlines:
            outline = "; ".join(self.outlines[name][:40])
            parts.append(f"[{name}] {self.briefs.get(name, '')}" + (f"\nTemas: {outline}" if outline else ""))
        return "\n\n".join(parts)
