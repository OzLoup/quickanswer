from pathlib import Path

HOME = Path.home() / ".quickanswer"
DOCS_DIR = HOME / "docs"
CACHE_DIR = HOME / "cache"
for d in (DOCS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"   # 14b = más preciso y más lento: qwen2.5:14b-instruct
CLAUDE_MODELS = {"claude-haiku": "claude-haiku-5-5", "claude-sonnet": "claude-sonnet-5-5"}
API_KEY_FILE = HOME / "api_key"      # clave de Anthropic (o variable de entorno ANTHROPIC_API_KEY)
EMBED_MODEL = "bge-m3"               # multilingüe: pregunta en ES encuentra texto en EN y viceversa
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"   # GPU de Apple; alternativa: mlx-community/whisper-medium-mlx

SAMPLE_RATE = 16000
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 4                      # fragmentos para el modelo local
TOP_K_CLAUDE = 8               # fragmentos para Claude si el documento no cabe completo
FULL_DOC_MAX_CHARS = 800_000   # ≈200k tokens: por debajo, Claude recibe los documentos completos (en caché)
