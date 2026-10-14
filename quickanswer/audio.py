"""Captura de dos flujos (micrófono = yo, BlackHole = interlocutor) con segmentación por silencios."""
import queue
from typing import Callable

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE

FRAME_S = 0.03
END_SILENCE_S = 0.5
MIN_SPEECH_S = 0.4
MAX_SEGMENT_S = 20.0


def list_inputs() -> list[dict]:
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
    ]


def find_system_device() -> int | None:
    for d in list_inputs():
        if "blackhole" in d["name"].lower():
            return d["index"]
    return None


def default_mic() -> int | None:
    try:
        idx = sd.default.device[0]
        return idx if idx is not None and idx >= 0 else None
    except Exception:
        return None


def _to_16k_mono(block: np.ndarray, rate: int) -> np.ndarray:
    mono = block.mean(axis=1) if block.ndim > 1 else block
    if rate == SAMPLE_RATE:
        return mono
    if rate % SAMPLE_RATE == 0:
        f = rate // SAMPLE_RATE
        n = len(mono) // f * f
        return mono[:n].reshape(-1, f).mean(axis=1)
    n_out = int(len(mono) * SAMPLE_RATE / rate)
    return np.interp(np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono)


class Segmenter:
    """VAD por energía con piso de ruido adaptativo. Emite segmentos float32 16 kHz."""

    def __init__(self, on_segment: Callable[[np.ndarray], None]):
        self.on_segment = on_segment
        self.noise = 0.003
        self.buf: list[np.ndarray] = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.in_speech = False
        self.pending = np.zeros(0, dtype=np.float32)
        self.frame = int(SAMPLE_RATE * FRAME_S)

    def feed(self, samples: np.ndarray):
        self.pending = np.concatenate([self.pending, samples.astype(np.float32)])
        while len(self.pending) >= self.frame:
            fr, self.pending = self.pending[:self.frame], self.pending[self.frame:]
            self._frame(fr)

    def _frame(self, fr: np.ndarray):
        rms = float(np.sqrt(np.mean(fr * fr)) + 1e-9)
        voiced = rms > max(0.006, self.noise * 3)
        if not voiced:
            self.noise = 0.98 * self.noise + 0.02 * rms
        if voiced:
            self.in_speech = True
            self.speech_frames += 1
            self.silence_frames = 0
            self.buf.append(fr)
        elif self.in_speech:
            self.buf.append(fr)
            self.silence_frames += 1
            if self.silence_frames * FRAME_S >= END_SILENCE_S:
                self._finish()
        if self.in_speech and len(self.buf) * FRAME_S >= MAX_SEGMENT_S:
            self._finish()

    def _finish(self):
        if self.speech_frames * FRAME_S >= MIN_SPEECH_S:
            self.on_segment(np.concatenate(self.buf))
        self.buf, self.speech_frames, self.silence_frames, self.in_speech = [], 0, 0, False


class Capture:
    """Abre un stream por fuente. on_segment(source, audio) con source in {'mic','system'}."""

    def __init__(self, on_segment: Callable[[str, np.ndarray], None]):
        self.on_segment = on_segment
        self.streams: list[sd.InputStream] = []

    def start(self, mic: int | None, system: int | None):
        for source, dev in (("mic", mic), ("system", system)):
            if dev is None:
                continue
            info = sd.query_devices(dev)
            rate = int(info["default_samplerate"])
            seg = Segmenter(lambda a, s=source: self.on_segment(s, a))
            q: queue.Queue = queue.Queue()

            def cb(indata, frames, t, status, seg=seg, rate=rate):
                seg.feed(_to_16k_mono(indata.copy(), rate))

            st = sd.InputStream(device=dev, channels=min(2, info["max_input_channels"]),
                                samplerate=rate, blocksize=int(rate * FRAME_S * 4), callback=cb)
            st.start()
            self.streams.append(st)

    def stop(self):
        for s in self.streams:
            s.stop(); s.close()
        self.streams = []
