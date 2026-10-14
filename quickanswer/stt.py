"""Voz a texto con Whisper en la GPU de Apple Silicon (MLX), con detector de voz previo."""
import mlx_whisper
import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .config import WHISPER_MODEL

_VAD = VadOptions(min_silence_duration_ms=500)


class Transcriber:
    def __init__(self):
        # Descarga el modelo la primera vez y lo deja cargado (calentamiento con 1 s de silencio).
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=WHISPER_MODEL)
        get_speech_timestamps(np.zeros(16000, dtype=np.float32), _VAD)

    def transcribe(self, audio, lang: str, hint: str = "") -> str:
        audio = audio.astype(np.float32)
        # Whisper inventa frases sobre ruido (teclado, respiración): solo transcribe si Silero detecta voz.
        if not get_speech_timestamps(audio, _VAD):
            return ""
        r = mlx_whisper.transcribe(
            audio, path_or_hf_repo=WHISPER_MODEL, language=lang,
            temperature=(0.0, 0.4), condition_on_previous_text=False,
            initial_prompt=f"Términos técnicos: {hint}." if hint else None,
        )
        # Descarta segmentos repetitivos o de baja confianza (típicas alucinaciones).
        ok = [s["text"].strip() for s in r["segments"]
              if s.get("compression_ratio", 0) <= 2.4 and s.get("avg_logprob", 0) >= -1.0]
        return " ".join(ok).strip()
