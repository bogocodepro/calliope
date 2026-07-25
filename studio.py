"""Calliope Studio — one UI for the whole voice lifecycle.

Tabs:
  • Speak  — type text, pick any voice (8 presets + your trained ones), speed/temp/model, play.
  • Voices — create & name unlimited custom voices, record a dataset for each, fine-tune
             on your GPU, watch progress, then use the trained voice in Speak.

Run:  ./run_studio.sh    then open  http://localhost:7860
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
from pedalboard import time_stretch    # high-quality Rubber Band speed-up (no echo)

from config import config
from services.orpheus_local import OrpheusTTS, ENGLISH_VOICES
from record_ui import SENTENCES, webm_to_wav24k

PORT = int(os.environ.get("STUDIO_PORT", "7860"))
ROOT = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(ROOT, "voice_dataset", "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

MODELS_DIR = os.path.expanduser("~/.local/share/calliope-models")
MODEL_FILES = {"q8": os.path.join(MODELS_DIR, "orpheus-q8.gguf"),
               "q4": os.path.join(MODELS_DIR, "orpheus-q4.gguf")}
PRESETS = [("tara", "f"), ("leah", "f"), ("jess", "f"), ("mia", "f"), ("zoe", "f"),
           ("leo", "m"), ("dan", "m"), ("zac", "m")]

# ---- load base engines (share one SNAC) ----
# Q8 only by default (best quality + frees VRAM so training can coexist);
# set STUDIO_LOAD_Q4=1 to also load Q4 for A/B comparison.
_want = ["q8"] + (["q4"] if os.environ.get("STUDIO_LOAD_Q4") == "1" else [])
_engines = {}
_shared_snac = None
for key in _want:
    path = MODEL_FILES[key]
    if os.path.exists(path):
        print(f"Loading base Orpheus {key.upper()}…", flush=True)
        eng = OrpheusTTS(model_path=path, voice="tara", device="cuda",
                         temperature=0.25, snac_model=_shared_snac)
        _shared_snac = eng.snac
        _engines[key] = eng
if not _engines:
    raise SystemExit("No base Orpheus GGUF found in " + MODELS_DIR)
DEFAULT_MODEL = "q8" if "q8" in _engines else "q4"

_lock = threading.Lock()               # base model (llama.cpp not thread-safe)
_custom_lock = threading.Lock()
_custom_active = {"name": None, "engine": None}   # single custom slot to bound VRAM


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", name.strip().lower().replace(" ", "_"))[:32]


def vdir(name: str) -> str:
    return os.path.join(VOICES_DIR, safe_name(name))


def read_status(d: str) -> dict:
    p = os.path.join(d, "status.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write_status(d: str, st: dict):
    json.dump(st, open(os.path.join(d, "status.json"), "w"))


def clip_count(d: str) -> int:
    c = os.path.join(d, "clips")
    return len([f for f in os.listdir(c) if f.endswith(".wav")]) if os.path.isdir(c) else 0


def write_meta(d: str):
    cd = os.path.join(d, "clips")
    rows = []
    if os.path.isdir(cd):
        for f in sorted(os.listdir(cd)):
            if f.endswith(".wav") and f[:-4].isdigit():
                rows.append([f"clips/{f}", SENTENCES[int(f[:-4])]])
    with open(os.path.join(d, "metadata.csv"), "w", newline="") as mf:
        csv.writer(mf).writerows(rows)


def evict_custom(name: str):
    with _custom_lock:
        if _custom_active["name"] == name:
            _custom_active.update(name=None, engine=None)


def list_custom() -> list:
    out = []
    for name in sorted(os.listdir(VOICES_DIR)):
        d = os.path.join(VOICES_DIR, name)
        if not os.path.isdir(d):
            continue
        st = read_status(d)
        out.append({"name": name, "state": st.get("state", "new"),
                    "clips": clip_count(d), "total": len(SENTENCES)})
    return out


def _speedup(audio, speed):
    """Speed up speech WITHOUT pitch change or echo, via Rubber Band (pedalboard)."""
    if abs(speed - 1.0) < 1e-3 or len(audio) < 512:
        return audio
    out = time_stretch(audio.reshape(1, -1), 24000, stretch_factor=float(speed),
                       high_quality=True)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def get_custom_engine(name: str):
    """Load base+adapter for a trained custom voice; keep one in VRAM at a time."""
    d = vdir(name); st = read_status(d)
    if st.get("state") != "ready" or not st.get("gguf"):
        raise RuntimeError(f"voice '{name}' is not trained yet")
    with _custom_lock:
        if _custom_active["name"] == name and _custom_active["engine"]:
            return _custom_active["engine"]
        _custom_active["engine"] = None  # evict previous (bounds VRAM to one custom voice)
        eng = OrpheusTTS(model_path=st["gguf"], voice=name, device="cuda",
                         temperature=0.25, snac_model=_shared_snac, finetuned=True)
        _custom_active.update(name=name, engine=eng)
        return eng


def synth_wav(text, voice, temp, speed, model) -> bytes:
    if voice in ENGLISH_VOICES:
        eng = _engines.get(model) or _engines[DEFAULT_MODEL]
        with _lock:
            eng.voice = voice
            eng.temperature = float(temp)
            audio = eng.synth_full(text)
    else:
        eng = get_custom_engine(voice)
        with _custom_lock:
            eng.temperature = float(temp)
            audio = eng.synth_full(text)
    audio = _speedup(audio, float(speed))
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1e-4:                       # normalize loudness (esp. quiet custom voices)
        audio = np.clip(audio * min(0.95 / peak, 20.0), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---- training job control ----
def start_training(name: str):
    d = vdir(name)
    if clip_count(d) < 20:
        raise RuntimeError("need at least 20 recorded clips to train")
    st = read_status(d)
    if st.get("state") == "training":
        raise RuntimeError("already training")
    write_status(d, {**st, "state": "training", "started": time.time()})
    log = open(os.path.join(d, "train.log"), "w")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser(
        "~/.local/palib/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", ""))
    # IMPORTANT: use the isolated training venv (unsloth/CUDA-13), not the inference venv.
    train_py = os.path.join(ROOT, ".venv-train", "bin", "python")
    subprocess.Popen([train_py, os.path.join(ROOT, "train_voice.py"), safe_name(name)],
                     stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)


def train_status(name: str) -> dict:
    d = vdir(name); st = read_status(d)
    logp = os.path.join(d, "train.log")
    tail = ""
    if os.path.exists(logp):
        with open(logp) as f:
            tail = "".join(f.readlines()[-14:])
    return {"state": st.get("state", "new"), "log": tail}


# ---- live call (conversation agent) control ----
_call = {"proc": None, "log": None, "voice": None}
CALL_LOG = os.path.join(ROOT, "voice_dataset", "call.log")


def ensure_ollama() -> bool:
    import urllib.request
    def up():
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False
    if up():
        return True
    ol = os.path.expanduser("~/.local/bin/ollama")
    if os.path.exists(ol):
        subprocess.Popen(["setsid", ol, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(1)
            if up():
                return True
    return False


def start_call(voice: str, persona: str):
    p = _call.get("proc")
    if p and p.poll() is None:
        raise RuntimeError("a call is already running")
    if not ensure_ollama():
        raise RuntimeError("brain (Ollama) not available — install it / run `ollama serve`")
    log = open(CALL_LOG, "w")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser(
        "~/.local/palib/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", ""))
    env["ORPHEUS_VOICE"] = voice
    env["PERSONA"] = persona if persona in ("casual", "sales") else "casual"
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "talk.py")],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    _call.update(proc=proc, log=CALL_LOG, voice=voice)


def stop_call():
    p = _call.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    _call["proc"] = None


def call_status() -> dict:
    p = _call.get("proc")
    running = bool(p and p.poll() is None)
    transcript, ready = [], False
    if os.path.exists(CALL_LOG):
        txt = open(CALL_LOG, errors="ignore").read()
        ready = "Calliope is live" in txt
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("🤖") or s.startswith("🗣️"):
                transcript.append(s)
    return {"running": running, "ready": ready and running,
            "voice": _call.get("voice"), "transcript": transcript[-40:]}


PAGE = open(os.path.join(ROOT, "studio.html")).read() if os.path.exists(
    os.path.join(ROOT, "studio.html")) else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _wav(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(p.query)
        if p.path == "/":
            body = (PAGE
                    .replace("%%PRESETS%%", json.dumps(PRESETS))
                    .replace("%%SENTENCES%%", json.dumps(SENTENCES))
                    .replace("%%MODELS%%", json.dumps(list(_engines.keys())))
                    .replace("%%DEFAULT%%", json.dumps(DEFAULT_MODEL))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p.path == "/api/voices":
            self._json({"presets": PRESETS, "custom": list_custom()})
            return
        if p.path == "/api/clip":
            name = q.get("voice", [""])[0]; i = int(q.get("i", ["-1"])[0])
            path = os.path.join(vdir(name), "clips", f"{i:03d}.wav")
            if not os.path.exists(path):
                self.send_error(404); return
            self._wav(open(path, "rb").read()); return
        if p.path == "/api/voice_detail":
            name = q.get("voice", [""])[0]; d = vdir(name)
            cd = os.path.join(d, "clips")
            rec = sorted(int(f[:-4]) for f in os.listdir(cd)
                         if f.endswith(".wav") and f[:-4].isdigit()) if os.path.isdir(cd) else []
            self._json({"state": read_status(d).get("state", "new"), "recorded": rec}); return
        if p.path == "/api/train_status":
            self._json(train_status(q.get("voice", [""])[0])); return
        if p.path == "/api/call_status":
            self._json(call_status()); return
        if p.path == "/api/synth":
            text = q.get("text", [""])[0].strip()
            if not text:
                self.send_error(400, "empty"); return
            try:
                wav = synth_wav(text, q.get("voice", ["tara"])[0],
                                float(q.get("temp", ["0.25"])[0]),
                                float(q.get("speed", ["1.15"])[0]),
                                q.get("model", [DEFAULT_MODEL])[0])
                self._wav(wav)
            except Exception as e:
                self.send_error(500, str(e))
            return
        self.send_error(404)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(p.query)
        if p.path == "/api/voice_create":
            name = safe_name(q.get("name", [""])[0])
            if not name:
                self.send_error(400, "bad name"); return
            if name in ENGLISH_VOICES:
                self.send_error(400, "name reserved"); return
            os.makedirs(os.path.join(vdir(name), "clips"), exist_ok=True)
            d = vdir(name)
            if not read_status(d):
                write_status(d, {"state": "new"})
            self._json({"ok": True, "name": name}); return
        if p.path == "/api/record":
            name = q.get("voice", [""])[0]; i = int(q.get("i", ["-1"])[0])
            d = vdir(name)
            if not os.path.isdir(d) or not (0 <= i < len(SENTENCES)):
                self.send_error(400, "bad"); return
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                pcm = webm_to_wav24k(data)
                if len(pcm) < 2400:
                    self.send_error(400, "too short"); return
                # normalize each clip to a consistent loudness so quiet mic levels
                # don't train a quiet/weak voice (recorder has auto-gain off)
                pf = pcm.astype(np.float32) / 32768.0
                pk = float(np.abs(pf).max())
                if pk > 1e-4:
                    pf = np.clip(pf * min(0.95 / pk, 30.0), -1.0, 1.0)
                pcm = (pf * 32767).astype("<i2")
                os.makedirs(os.path.join(d, "clips"), exist_ok=True)
                sf.write(os.path.join(d, "clips", f"{i:03d}.wav"), pcm, 24000, subtype="PCM_16")
                write_meta(d)
                st = read_status(d)
                if st.get("state") in ("new", None):
                    write_status(d, {**st, "state": "recording"})
                self._json({"ok": True, "clips": clip_count(d)})
            except Exception as e:
                self.send_error(500, str(e))
            return
        if p.path == "/api/train":
            try:
                start_training(q.get("voice", [""])[0])
                self._json({"ok": True})
            except Exception as e:
                self.send_error(400, str(e))
            return
        if p.path == "/api/call_start":
            try:
                start_call(q.get("voice", ["dan"])[0], q.get("persona", ["casual"])[0])
                self._json({"ok": True})
            except Exception as e:
                self.send_error(400, str(e))
            return
        if p.path == "/api/call_stop":
            stop_call(); self._json({"ok": True}); return
        if p.path == "/api/voice_delete":
            name = q.get("voice", [""])[0]; d = vdir(name)
            if os.path.isdir(d):
                evict_custom(name)
                shutil.rmtree(d, ignore_errors=True)
            self._json({"ok": True}); return
        if p.path == "/api/clip_delete":
            name = q.get("voice", [""])[0]; i = int(q.get("i", ["-1"])[0]); d = vdir(name)
            fp = os.path.join(d, "clips", f"{i:03d}.wav")
            if os.path.exists(fp):
                os.remove(fp); write_meta(d)
            if clip_count(d) == 0 and read_status(d).get("state") == "recording":
                write_status(d, {"state": "new"})
            self._json({"ok": True, "clips": clip_count(d)}); return
        if p.path == "/api/voice_rename":
            name = q.get("voice", [""])[0]; to = safe_name(q.get("to", [""])[0])
            d = vdir(name)
            if not os.path.isdir(d):
                self.send_error(404, "no such voice"); return
            if not to or to in ENGLISH_VOICES or os.path.isdir(vdir(to)):
                self.send_error(400, "bad or taken name"); return
            if read_status(d).get("state") == "ready":
                self.send_error(400, "trained voice — delete & re-record to rename (speaker is baked in)"); return
            evict_custom(name)
            os.rename(d, vdir(to)); write_meta(vdir(to))
            self._json({"ok": True, "name": to}); return
        self.send_error(404)


if __name__ == "__main__":
    print(f"Studio ready — open http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
