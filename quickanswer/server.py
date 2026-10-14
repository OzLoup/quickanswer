import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import audio, llm
from .config import CLAUDE_MODELS, DOCS_DIR, FULL_DOC_MAX_CHARS, TOP_K_CLAUDE
from .qa import build_claude, build_messages, claude_system, is_question
from .rag import Index

app = FastAPI()
STATIC = Path(__file__).parent / "static"


class State:
    lang = "es"
    provider = "local"   # "local" | "claude-haiku" | "claude-sonnet"
    running = False
    index = Index()
    clients: set[WebSocket] = set()
    transcript: list[tuple[float, str, str]] = []   # (t, who, text)
    answer_task: asyncio.Task | None = None
    capture: audio.Capture | None = None
    transcriber = None
    seg_q: "asyncio.Queue | None" = None
    loop: asyncio.AbstractEventLoop | None = None
    last_q, last_q_t = "", 0.0   # última pregunta lanzada, para detectar continuaciones


S = State()


async def broadcast(msg: dict):
    dead = []
    for ws in S.clients:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        S.clients.discard(ws)


def emit(msg: dict):
    """Seguro desde hilos."""
    if S.loop:
        asyncio.run_coroutine_threadsafe(broadcast(msg), S.loop)


@app.on_event("startup")
async def _startup():
    S.loop = asyncio.get_running_loop()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/status")
async def status():
    ollama = await llm.check()
    return {
        "ollama": ollama, "devices": audio.list_inputs(),
        "system_device": audio.find_system_device(), "mic_device": audio.default_mic(),
        "docs": list(S.index.outlines), "running": S.running, "lang": S.lang,
        "provider": S.provider, "has_key": bool(llm.get_api_key()),
    }


# ---------- documentos ----------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dest = DOCS_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    asyncio.create_task(_index_doc(dest, S.lang))
    return {"ok": True, "name": dest.name}


async def _index_doc(path: Path, lang: str):
    loop = asyncio.get_running_loop()
    prog = lambda d, t: emit({"type": "doc_progress", "name": path.name, "done": d, "total": t})
    try:
        await loop.run_in_executor(None, lambda: S.index.add_document(path, prog))
        emit({"type": "doc_ready", "name": path.name, "chunks": len(S.index.chunks), "brief": False})
        _schedule_prewarm()
        sample = S.index.sample_text(path.name)
        outline = "; ".join(S.index.outlines.get(path.name, [])[:40])
        brief = await llm.chat_once([
            {"role": "system", "content": f"Resume documentos en {'español' if lang == 'es' else 'English'}, máximo 6 líneas: tema, propósito, secciones y datos clave."},
            {"role": "user", "content": f"Archivo: {path.name}\nTemas: {outline}\n\nMuestras del contenido:\n{sample}"},
        ], num_ctx=8192)
        S.index.briefs[path.name] = brief.strip()
        emit({"type": "doc_ready", "name": path.name, "chunks": len(S.index.chunks), "brief": True})
    except Exception as e:
        emit({"type": "error", "text": f"{path.name}: {e}"})


@app.post("/api/provider")
async def set_provider(body: dict):
    if body.get("api_key"):
        llm.set_api_key(body["api_key"])
    prov = body.get("provider", "local")
    if prov != "local" and prov not in CLAUDE_MODELS:
        return {"ok": False, "error": "Proveedor desconocido"}
    if prov != "local" and not llm.get_api_key():
        return {"ok": False, "error": "Falta la clave de API de Anthropic."}
    S.provider = prov
    _schedule_prewarm()
    return {"ok": True, "provider": prov}


def _claude_corpus() -> str:
    corpus = S.index.full_corpus()
    return corpus if len(corpus) <= FULL_DOC_MAX_CHARS else ""


def _schedule_prewarm():
    if S.provider in CLAUDE_MODELS and S.loop:
        S.loop.create_task(_prewarm(S.provider, S.lang))


async def _prewarm(provider: str, lang: str):
    """Sube los documentos a la caché de Claude antes de la primera pregunta (y abre la conexión)."""
    corpus = _claude_corpus()
    if not corpus:
        return
    try:
        u = await llm.claude_prewarm(provider, claude_system(lang, corpus))
        tokens = (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
        emit({"type": "status", "text": f"Claude listo: documentos en caché ({tokens:,} tokens)"})
    except Exception as e:
        emit({"type": "error", "text": f"No se pudo preparar Claude: {e}"})


# ---------- sesión ----------

@app.post("/api/start")
async def start(body: dict):
    if S.running:
        return {"ok": True}
    S.lang = body.get("lang", "es")
    S.transcript.clear()
    S.seg_q = asyncio.Queue()
    if S.transcriber is None:
        emit({"type": "status", "text": "Cargando modelo de voz (primera vez tarda)…"})
        from .stt import Transcriber
        S.transcriber = await asyncio.get_running_loop().run_in_executor(None, Transcriber)
    if S.provider == "local":
        asyncio.create_task(llm.warmup())
    else:
        _schedule_prewarm()

    def on_segment(source: str, a):
        S.loop.call_soon_threadsafe(S.seg_q.put_nowait, (source, a))

    mic = body.get("mic", audio.default_mic())
    system = body.get("system", audio.find_system_device())
    S.capture = audio.Capture(on_segment)
    try:
        S.capture.start(mic, system)
    except Exception as e:
        return {"ok": False, "error": f"No se pudo abrir el audio: {e}"}
    S.running = True
    asyncio.create_task(_transcribe_loop())
    await broadcast({"type": "status", "text": f"Escuchando ({S.lang.upper()})", "running": True})
    return {"ok": True, "system_found": system is not None}


@app.post("/api/stop")
async def stop():
    S.running = False
    if S.capture:
        S.capture.stop()
    if S.seg_q:
        S.seg_q.put_nowait(None)
    await broadcast({"type": "status", "text": "Detenido", "running": False})
    return {"ok": True}


async def _transcribe_loop():
    loop = asyncio.get_running_loop()
    while S.running:
        item = await S.seg_q.get()
        if item is None:
            break
        source, a = item
        text = await loop.run_in_executor(None, S.transcriber.transcribe, a, S.lang, S.index.vocab_hint())
        if len(text) < 2:
            continue
        who = "me" if source == "mic" else "them"
        S.transcript.append((time.time(), who, text))
        await broadcast({"type": "transcript", "who": who, "text": text})
        if who == "them":
            _maybe_answer(text)


def _maybe_answer(text: str):
    """Responde con el bloque completo de la pregunta, aunque el jurado haya hecho pausas a mitad."""
    block = _last_them_block(max_gap=4)
    continues = bool(S.last_q) and block.startswith(S.last_q) and block != S.last_q and time.time() - S.last_q_t < 8
    if is_question(text, S.lang) or continues:
        _launch_answer(block)


def _recent_context(n: int = 6) -> list[str]:
    return [f"{'Presentador' if w == 'me' else 'Interlocutor'}: {t}" for _, w, t in S.transcript[-n:]]


def _last_them_block(max_gap: float = 15) -> str:
    """Últimas intervenciones seguidas del interlocutor (con huecos de hasta max_gap segundos)."""
    block, last_t = [], None
    for t, who, text in reversed(S.transcript):
        if who != "them":
            if block:
                break
            continue
        if last_t is not None and last_t - t > max_gap:
            break
        block.insert(0, text)
        last_t = t
    return " ".join(block)


def _launch_answer(question: str):
    if S.answer_task and not S.answer_task.done():
        S.answer_task.cancel()
    S.last_q, S.last_q_t = question, time.time()
    S.answer_task = asyncio.create_task(_answer(question))


async def _safe_search(question: str, k: int):
    try:
        return await S.index.search(question, k=k)
    except Exception:
        return []   # sin Ollama no hay fuentes, pero Claude responde igual


async def _answer(question: str):
    aid = int(time.time() * 1000)
    await broadcast({"type": "answer_start", "id": aid, "question": question})
    try:
        ctx = _recent_context()
        if S.provider == "local":
            excerpts = await S.index.search(question)
            stream = llm.chat_stream(build_messages(S.lang, question, ctx, S.index.brief(), excerpts))
            sources = asyncio.sleep(0, excerpts)
        elif corpus := _claude_corpus():
            # Documento completo en caché: Claude arranca ya; la búsqueda solo sirve para mostrar fuentes.
            system, msgs = build_claude(S.lang, question, ctx, corpus=corpus)
            stream = llm.claude_stream(S.provider, system, msgs)
            sources = asyncio.create_task(_safe_search(question, 3))
        else:
            excerpts = await S.index.search(question, k=TOP_K_CLAUDE)
            system, msgs = build_claude(S.lang, question, ctx, excerpts=excerpts)
            stream = llm.claude_stream(S.provider, system, msgs)
            sources = asyncio.sleep(0, excerpts)
        async for piece in stream:
            await broadcast({"type": "answer_delta", "id": aid, "text": piece})
        excerpts = await sources
        await broadcast({"type": "answer_end", "id": aid, "sources": sorted({c.loc.rstrip(" ·") for c in excerpts[:3]})})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        await broadcast({"type": "error", "text": f"Error al responder: {e}"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    S.clients.add(websocket)
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            if msg["type"] == "answer_last":
                q = _last_them_block()
                if q:
                    _launch_answer(q)
                else:
                    await websocket.send_text(json.dumps({"type": "error", "text": "Aún no hay nada del interlocutor."}))
            elif msg["type"] == "ask":
                _launch_answer(msg["text"])
    except WebSocketDisconnect:
        S.clients.discard(websocket)
