#!/usr/bin/env bash
# Instalación para macOS (Apple Silicon). Ejecutar una vez: ./setup_mac.sh
set -e
cd "$(dirname "$0")"

command -v brew >/dev/null || { echo "Instala Homebrew primero: https://brew.sh"; exit 1; }

echo "==> Ollama (LLM local)"
command -v ollama >/dev/null || brew install ollama
brew services start ollama >/dev/null 2>&1 || true
until curl -s localhost:11434/api/tags >/dev/null; do sleep 1; done

echo "==> Modelos (≈6 GB la primera vez)"
ollama pull qwen2.5:7b-instruct   # respuestas
ollama pull bge-m3                # búsqueda en documentos ES/EN

echo "==> Audio del sistema (BlackHole) — pedirá tu contraseña"
brew list --cask blackhole-2ch >/dev/null 2>&1 || brew install --cask blackhole-2ch

echo "==> Entorno Python"
command -v uv >/dev/null || brew install uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

cat <<'MSG'

Instalación lista. Falta UN paso manual (una sola vez) para que la app oiga a tu interlocutor:
  1. Abre "Configuración de Audio MIDI" > "+" > "Crear dispositivo de salida múltiple".
  2. Marca tus audífonos (o bocinas) Y "BlackHole 2ch".
  3. En Zoom/Teams > Ajustes de audio > Altavoz: elige ese dispositivo de salida múltiple.
Usa audífonos: si usas bocinas, tu micrófono recogerá al interlocutor y saldrá duplicado.

Arrancar:  .venv/bin/python run.py
MSG
