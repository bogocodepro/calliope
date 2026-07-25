"""Calliope — self-contained real-time voice conversation loop.

Pipeline (no Pipecat, no PyAudio — fewer moving parts, runs today):

    mic (sounddevice) -> Silero VAD endpointing -> faster-whisper STT
        -> LLM (Ollama or Gemini) -> Chatterbox TTS -> speakers (sounddevice)

Chatterbox (Resemble AI) is the voice: top-tier emotional/expressive open TTS,
single package. Barge-in: while the agent speaks, the mic keeps listening; if you
start talking it stops and listens.

Launch via run.sh (sets LD_LIBRARY_PATH for the user-space PortAudio):
    ./run.sh
Ctrl+C to quit.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

import numpy as np

# Preload user-space PortAudio (installed without sudo) before importing sounddevice.
try:  # pragma: no cover
    import ctypes

    _pa = os.path.expanduser("~/.local/palib/usr/lib/x86_64-linux-gnu/libportaudio.so.2")
    if os.path.exists(_pa):
        ctypes.CDLL(_pa, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import sounddevice as sd
import requests
from loguru import logger
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

from config import config


def speedup(audio: np.ndarray, speed: float) -> np.ndarray:
    """Speed up speech WITHOUT changing pitch (WSOLA — clean on voice, no phasiness)."""
    if speed == 1.0 or len(audio) < 512:
        return audio
    reader = ArrayReader(audio.reshape(1, -1))
    writer = ArrayWriter(channels=1)
    tsm = wsola(channels=1, speed=speed)
    tsm.run(reader, writer)
    return writer.data.flatten().astype(np.float32)

# Line-buffer stdout so transcripts show up promptly in logs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def configure_audio():
    """Route through PulseAudio (not raw ALSA) to avoid buffer underruns/choppy audio."""
    pulse_idx = None
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == "pulse":
            pulse_idx = i
            break
    if pulse_idx is not None:
        sd.default.device = (pulse_idx, pulse_idx)
        logger.info(f"Audio routed through PulseAudio (device {pulse_idx})")
    sd.default.latency = ("high", "high")

# ---- Audio constants ----
STT_SR = 16000          # whisper / VAD sample rate
VAD_FRAME = 512         # samples per VAD step at 16 kHz (~32 ms)
END_SILENCE_S = 0.7     # trailing silence that ends the user's turn
MIN_SPEECH_S = 0.3      # ignore blips shorter than this
BARGE_SPEECH_S = 0.4    # sustained speech during playback => interrupt

CASUAL_SYSTEM = (
    "You're on a casual one-on-one phone call, just chatting. Talk like a normal, "
    "relaxed person — NOT a presenter or performer. Keep replies short: usually one "
    "sentence, occasionally two. Use plain everyday words and contractions. It's fine "
    "to be low-key and a little unremarkable, like real small talk. Do NOT use "
    "exclamation marks, emoji, lists, markdown, or stage directions, and don't narrate "
    "your emotions. Say numbers as words. Just respond naturally to what they said."
)


class MicStream:
    """Persistent 16 kHz mono mic capture pushing frames onto a queue."""

    def __init__(self):
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=STT_SR, channels=1, dtype="float32",
            blocksize=VAD_FRAME, callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()

    def read(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class TTS:
    """Orpheus wrapper — the emotional voice. Streams float32 chunks at self.sr."""

    def __init__(self, device: str):
        from services.orpheus_local import OrpheusTTS

        self.engine = OrpheusTTS(
            model_path=config.orpheus_model_path,
            voice=config.orpheus_voice,
            device=device,
            temperature=config.orpheus_temperature,
        )
        self.sr = self.engine.sr

    def stream(self, text: str):
        return self.engine.stream(text)


def load_models():
    logger.info("Loading Silero VAD…")
    from silero_vad import load_silero_vad
    import torch

    vad = load_silero_vad()

    def vad_prob(frame: np.ndarray) -> float:
        with torch.no_grad():
            return vad(torch.from_numpy(frame), STT_SR).item()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Whisper ({config.whisper_model}) on {config.whisper_device}…")
    from faster_whisper import WhisperModel

    try:
        stt = WhisperModel(config.whisper_model, device=config.whisper_device,
                           compute_type=config.whisper_compute_type)
    except Exception as e:
        logger.warning(f"Whisper on GPU failed ({e}); using CPU int8")
        stt = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")

    logger.info("Loading Orpheus TTS…")
    tts = TTS(dev)
    return vad_prob, stt, tts


def transcribe(stt, audio: np.ndarray) -> str:
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return "".join(s.text for s in segments).strip()


def ask_ollama(messages) -> str:
    url = config.ollama_base_url.replace("/v1", "") + "/api/chat"
    r = requests.post(url, json={"model": config.ollama_model,
                                 "messages": messages, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def ask_gemini(messages) -> str:
    import google.generativeai as genai

    genai.configure(api_key=config.gemini_api_key)
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    model = genai.GenerativeModel(config.gemini_model, system_instruction=sys_msg)
    history = [{"role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]]}
               for m in messages if m["role"] != "system"]
    return model.generate_content(history).text.strip()


def ask_brain(messages) -> str:
    return ask_gemini(messages) if config.brain == "gemini" else ask_ollama(messages)


def listen(mic: MicStream, vad_prob) -> np.ndarray | None:
    """Block until the user speaks a full utterance; return 16 kHz float32 audio."""
    buf: list[np.ndarray] = []
    speaking = False
    silence = 0.0
    speech = 0.0
    frame_dur = VAD_FRAME / STT_SR
    while True:
        frame = mic.read(timeout=5.0)
        if frame is None:
            continue
        p = vad_prob(frame)
        if p > 0.5:
            speaking = True
            speech += frame_dur
            silence = 0.0
            buf.append(frame)
        elif speaking:
            silence += frame_dur
            buf.append(frame)
            if silence >= END_SILENCE_S:
                if speech >= MIN_SPEECH_S:
                    return np.concatenate(buf)
                buf, speaking, silence, speech = [], False, 0.0, 0.0


def speak(tts: TTS, mic: MicStream, vad_prob, text: str) -> bool:
    """Generate the whole reply, speed it up (WSOLA, pitch kept), then play it with
    barge-in. Returns True if the user interrupted. (WSOLA needs the full utterance,
    so we synth-then-play rather than stream; replies are short so latency is small.)"""
    interrupted = threading.Event()
    stopping = threading.Event()

    def watch_barge_in():
        run = 0.0
        frame_dur = VAD_FRAME / STT_SR
        while not interrupted.is_set() and not stopping.is_set():
            frame = mic.read(timeout=0.2)
            if frame is None:
                continue
            if vad_prob(frame) > 0.6:
                run += frame_dur
                if run >= BARGE_SPEECH_S:
                    interrupted.set()
                    return
            else:
                run = 0.0

    mic.drain()
    watcher = threading.Thread(target=watch_barge_in, daemon=True)
    watcher.start()

    # 1) generate full reply (user can interrupt during this silence)
    chunks = []
    for chunk in tts.stream(text):
        if interrupted.is_set():
            break
        chunks.append(np.asarray(chunk, dtype=np.float32))

    if interrupted.is_set() or not chunks:
        stopping.set()
        watcher.join(timeout=0.3)
        return interrupted.is_set()

    # 2) speed up (pitch preserved)
    audio = speedup(np.concatenate(chunks), config.orpheus_speed)

    # 3) play with barge-in
    pos = {"i": 0}

    def callback(outdata, frames, time_info, status):
        if interrupted.is_set():
            outdata.fill(0)
            raise sd.CallbackStop
        i = pos["i"]
        seg = audio[i:i + frames]
        n = len(seg)
        outdata[:n, 0] = seg
        if n < frames:
            outdata[n:, 0] = 0.0
            pos["i"] = len(audio)
            raise sd.CallbackStop
        pos["i"] = i + frames

    stream = sd.OutputStream(samplerate=tts.sr, channels=1, dtype="float32",
                             callback=callback, latency="high")
    stream.start()
    try:
        while stream.active and not interrupted.is_set():
            time.sleep(0.05)
    finally:
        stream.stop()
        stream.close()
        stopping.set()
        watcher.join(timeout=0.3)
    return interrupted.is_set()


def main():
    config.validate()
    persona = os.getenv("PERSONA", "casual")
    if persona == "sales":
        from prompts.sales_agent import build_system_prompt
        system = build_system_prompt()
    else:
        system = CASUAL_SYSTEM

    configure_audio()
    vad_prob, stt, tts = load_models()
    messages = [{"role": "system", "content": system}]

    print("\n" + "=" * 60)
    print("  Calliope is live. Put your headset on and just start talking.")
    print("  It'll reply out loud. Talk over it to interrupt. Ctrl+C to quit.")
    print("=" * 60 + "\n")

    with MicStream() as mic:
        opener = ("Hey, just so you know, I'm an AI voice. Anyway... "
                  "what've you been up to today?")
        messages.append({"role": "assistant", "content": opener})
        print(f"🤖 {opener}")
        speak(tts, mic, vad_prob, opener)

        while True:
            mic.drain()
            audio = listen(mic, vad_prob)
            if audio is None:
                continue
            t0 = time.time()
            user_text = transcribe(stt, audio)
            if not user_text:
                continue
            print(f"🗣️  {user_text}")
            messages.append({"role": "user", "content": user_text})
            reply = ask_brain(messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"🤖 {reply}   ({time.time() - t0:.1f}s)")
            speak(tts, mic, vad_prob, reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
