# Calliope

**A local, free, private voice AI toolkit.** Generate speech in a lineup of expressive
voices, clone **your own voice** by recording ~15 minutes of audio and fine-tuning it on
your GPU, and run a real-time voice agent that listens and talks back — all on your own
machine, offline, with no subscriptions or API keys.

Built on open models: **Orpheus** (emotional TTS), **SNAC** (audio codec),
**faster-whisper** (speech-to-text), **Silero VAD**, **Ollama** (local LLM brain), and
**Unsloth** (fine-tuning).

> **Ethics & legality:** This can power automated phone calls. AI voices in calls are
> regulated (US FCC/TCPA, EU AI Act, various states) — you generally must **disclose
> that the caller is AI**. The included sales persona does this by default; keep it that
> way. Clone **your own** voice, or a voice you have explicit permission to use. Don't
> impersonate real people.

---

## What's inside

| Tool | Command | What it does |
|------|---------|--------------|
| **Studio** (main) | `./run_studio.sh` → http://localhost:7860 | Web UI: type text and hear any voice, and create/record/train/manage **custom voices** |
| **Live agent** | `./run.sh` | Real-time headset conversation: you talk, it listens (Whisper), thinks (Ollama), and replies in a chosen voice |
| **Recorder** (standalone) | `./run_record.sh` → http://localhost:7861 | Just the voice-dataset recorder (also built into the Studio) |

The **Studio is where you'll spend your time.** The live agent is a proof-of-concept
phone-style assistant.

---

## How it works

```
 TEXT ─▶ Orpheus (GGUF on GPU, via llama.cpp) ─▶ audio tokens ─▶ SNAC decode ─▶ 24kHz speech

 Live agent adds:  mic ─▶ Silero VAD ─▶ faster-whisper (STT) ─▶ Ollama (LLM) ─▶ [above] ─▶ speakers
```

Custom voices are made by **fine-tuning Orpheus (LoRA)** on your recordings, then
exporting a per-voice GGUF the Studio loads on demand.

---

## Requirements

- **GPU:** NVIDIA with **~10 GB+ VRAM** (developed on a 16 GB RTX 5000 Ada). CUDA 12+ driver.
- **OS:** Linux (tested on Ubuntu 22.04). macOS can *run* inference with tweaks (Metal),
  but **fine-tuning needs CUDA** — see [Notes](#notes--limitations).
- **Python 3.10+**, `git`, `curl`.
- **~15 GB disk** for the models, plus ~3.3 GB per trained voice.
- For the **live agent only:** system PortAudio + [Ollama](https://ollama.com).

---

## Installation

```bash
git clone https://github.com/bogocodepro/calliope.git
cd calliope
```

### 1) Inference environment (the Studio + agent)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch — MUST be the CUDA 12.4 build (do this first)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# llama.cpp (CUDA prebuilt wheel — no compiling)
pip install llama-cpp-python==0.3.34 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# everything else
pip install -r requirements.txt
```

### 2) Download the voice model (~3.8 GB)

```bash
mkdir -p ~/.local/share/calliope-models
curl -fL https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf/resolve/main/Orpheus-3b-FT-Q8_0.gguf \
  -o ~/.local/share/calliope-models/orpheus-q8.gguf
```

(The SNAC audio decoder downloads itself automatically on first run.)

### 3) Run the Studio

```bash
./run_studio.sh
# open http://localhost:7860
```

That's it for text-to-speech and voice cloning. The steps below are only for extras.

### 4) (Optional) Voice fine-tuning environment

Fine-tuning uses **Unsloth**, which needs a **separate** venv (it pulls a different CUDA
build that must not mix with the inference venv):

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

### 5) (Optional) Live headset agent

Needs system PortAudio and a local LLM brain:

```bash
sudo apt install portaudio19-dev ffmpeg          # Debian/Ubuntu
curl -fsSL https://ollama.com/install.sh | sh    # Ollama
ollama pull llama3.2:3b
./run.sh                                          # PERSONA=sales ./run.sh  for the sales script
```

No `sudo`? See [Troubleshooting](#troubleshooting) for a no-root PortAudio trick.

---

## Using the Studio

Open **http://localhost:7860**.

### Speak tab
- Type text (inline emotion tags work: `<laugh>`, `<sigh>`, `<chuckle>`, `<gasp>`).
- Pick a **voice** — 8 built-in presets (`tara, leah, jess, mia, zoe, leo, dan, zac`)
  plus any voices **you've trained**.
- **Speed** (pitch-preserving) and **Temperature** (lower = calmer) sliders.
- **Generate & Play**, and **⬇ Download** the WAV.

### Voices tab — clone your own voice
1. **Create** a voice and name it (e.g. `my_voice`).
2. **Record** the sentences it shows you (Space = record/stop, ←/→ = navigate).
   Aim for **50+** clips. **Read in the exact calm, one-to-one tone you want the agent
   to have** — it learns your delivery. Quiet room, steady mic distance.
   Delete/re-record any bad takes.
3. **Train this voice** — runs a LoRA fine-tune on your GPU (~5 min training + a one-time
   6 GB base-model download for the first export). Progress shows live.
4. When it finishes, your voice appears in the **Speak** tab. Full delete/rename controls
   are provided.

---

## Configuration

Copy `.env.example` to `.env` to override defaults (voice, speed, brain model, sales
persona, etc.). All values are optional. See `config.py` for the full list.

---

## Project structure

```
studio.py / studio.html   # the Studio web app (Speak + Voices)
run_studio.sh             # launch the Studio
train_voice.py            # LoRA fine-tune + GGUF export (runs in .venv-train)
record_ui.py              # dataset recorder (also imported by the Studio)
talk.py / run.sh          # live headset conversation agent
services/orpheus_local.py # Orpheus GGUF + SNAC in-process TTS engine
prompts/sales_agent.py    # sales persona (with mandatory AI disclosure)
config.py                 # settings, read from .env
voice_dataset/            # YOUR recordings + trained voices (gitignored, never published)
```

---

## Troubleshooting

- **`torch` / CUDA errors, `libcusparse`/`libnvJitLink` mismatch** — you have the wrong
  torch build. Reinstall exactly `torch==2.6.0 torchaudio==2.6.0` from the `cu124` index.
  Never let the training venv's packages leak into the inference venv (keep them separate).
- **Training download hangs at a round number (e.g. 1 GB)** — HuggingFace's "xet"
  transfer stalling. Already handled (`HF_HUB_DISABLE_XET=1` is set in `train_voice.py`).
- **Choppy audio in the live agent (ALSA underruns)** — it routes through PulseAudio and
  uses a callback stream to avoid this; ensure PulseAudio/PipeWire is running.
- **No `sudo` for PortAudio (live agent):** download the `.deb` and point at it without
  root:
  ```bash
  apt-get download libportaudio2
  dpkg -x libportaudio2_*.deb ~/.local/palib
  ln -sf libportaudio.so.2 ~/.local/palib/usr/lib/x86_64-linux-gnu/libportaudio.so
  export LD_LIBRARY_PATH=~/.local/palib/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
- **Out of VRAM during training** — the Studio loads only Q8 by default to leave room;
  close other GPU apps. Training uses 4-bit QLoRA to stay light.
- **Cracky/robotic cloned voice** — usually the dataset: record more clips, speak at a
  natural (not slow) pace, and review takes.

---

## Notes & limitations

- **macOS / Apple Silicon:** inference can run via Metal (llama.cpp Metal + Torch MPS),
  but **fine-tuning (Unsloth) is CUDA-only** — train on an NVIDIA GPU (or cloud), then
  copy the resulting voice GGUF to the Mac to use.
- **Telephony** (dialing real phone numbers) is **not** included — this is the local
  voice engine. Wiring it to Twilio/Telnyx/SIP is a separate step (and where costs enter).

---

## Credits

[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) ·
[SNAC](https://github.com/hubertsiuzdak/snac) ·
[llama.cpp](https://github.com/ggerganov/llama.cpp) ·
[Unsloth](https://github.com/unslothai/unsloth) ·
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[Silero VAD](https://github.com/snakers4/silero-vad) ·
[Ollama](https://ollama.com)

## License

[MIT](LICENSE) © 2026 bogocodepro
