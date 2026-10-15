"""Clientes de LLM: Ollama (local, sin tokens) y Claude (API de Anthropic)."""
import json
import os

import anthropic
import httpx

from .config import API_KEY_FILE, CHAT_MODEL, CLAUDE_MODELS, EMBED_MODEL, OLLAMA_URL


async def check() -> dict:
    """Verifica que Ollama corre y que los modelos están descargados."""
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            tags = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
    except Exception:
        return {"ok": False, "error": "Ollama no responde en localhost:11434"}
    names = {m["name"] for m in tags.get("models", [])}
    have = lambda m: any(n == m or n.startswith(m + ":") or n.split(":")[0] == m.split(":")[0] and ":" not in m for n in names)
    missing = [m for m in (CHAT_MODEL, EMBED_MODEL) if not have(m)]
    return {"ok": not missing, "error": f"Faltan modelos: {', '.join(missing)}" if missing else ""}


async def warmup():
    async with httpx.AsyncClient(timeout=120) as c:
        await c.post(f"{OLLAMA_URL}/api/generate", json={"model": CHAT_MODEL, "keep_alive": "60m"})


async def unload_local():
    """Saca el modelo local de la memoria (≈5 GB) cuando se usa Claude."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{OLLAMA_URL}/api/generate", json={"model": CHAT_MODEL, "keep_alive": 0})
    except Exception:
        pass


def embed_sync(texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts, "keep_alive": "60m"}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]


async def embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts, "keep_alive": "60m"})
        r.raise_for_status()
        return r.json()["embeddings"]


def get_api_key() -> str:
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def set_api_key(key: str):
    API_KEY_FILE.write_text(key.strip())
    API_KEY_FILE.chmod(0o600)


_claude: anthropic.AsyncAnthropic | None = None
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _client() -> anthropic.AsyncAnthropic:
    """Un solo cliente reutilizado: la conexión HTTPS queda abierta entre preguntas."""
    global _claude
    key = get_api_key()
    if not key:
        raise RuntimeError("Falta la clave de API de Anthropic (pégala en la interfaz).")
    if _claude is None or _claude.api_key != key:
        _claude = anthropic.AsyncAnthropic(api_key=key, max_retries=1, timeout=60)
    return _claude


def _claude_params(provider: str, system: list[dict], messages: list[dict], max_tokens: int) -> dict:
    params = {"model": CLAUDE_MODELS[provider], "max_tokens": max_tokens, "system": system, "messages": messages}
    if "haiku" in provider:
        params["thinking"] = {"type": "disabled"}   # si no, el razonamiento consume max_tokens y la respuesta sale vacía
    else:
        params["thinking"] = {"type": "between_tools"}   # Sonnet 5.5: sin razonamiento previo = primera palabra antes
        params["betas"] = [_FALLBACK_BETA]
        params["fallbacks"] = "default"   # si Sonnet declina, el servidor reintenta en otro modelo
    return params


def _messages_api(provider: str):
    c = _client()
    return c.messages if "haiku" in provider else c.beta.messages


async def claude_stream(provider: str, system: list[dict], messages: list[dict]):
    params = _claude_params(provider, system, messages, max_tokens=600)
    async with _messages_api(provider).stream(**params) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
    if final.stop_reason == "refusal":
        raise RuntimeError("Claude declinó responder esta pregunta; prueba con el modelo local.")


async def claude_prewarm(provider: str, system: list[dict]):
    """Escribe el documento en la caché (max_tokens=0) y abre la conexión antes de la primera pregunta."""
    params = _claude_params(provider, system, [{"role": "user", "content": "."}], max_tokens=0)
    r = await _messages_api(provider).create(**params)
    return r.usage


async def chat_stream(messages: list[dict], temperature: float = 0.2, num_ctx: int = 4096):
    async for piece in _ollama_stream(messages, temperature, num_ctx):
        yield piece


async def _ollama_stream(messages: list[dict], temperature: float, num_ctx: int):
    payload = {
        "model": CHAT_MODEL, "messages": messages, "stream": True, "keep_alive": "60m",
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if piece := data.get("message", {}).get("content"):
                    yield piece
                if data.get("done"):
                    break


async def chat_once(messages: list[dict], **kw) -> str:
    return "".join([p async for p in chat_stream(messages, **kw)])
