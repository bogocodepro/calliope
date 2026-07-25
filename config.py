"""Central configuration for Calliope, loaded from environment / .env.

One place to switch the brain (Ollama <-> Gemini), pick models, and set the
editable sales script. Everything has a default so the app runs out of the box
once the local model servers are up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class Config:
    # ---- Brain ----
    brain: str = _get("BRAIN", "ollama").lower()  # "ollama" | "gemini"

    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.2:3b")

    gemini_api_key: str = _get("GEMINI_API_KEY", "")
    gemini_model: str = _get("GEMINI_MODEL", "gemini-2.0-flash")

    # ---- STT ----
    whisper_model: str = _get("WHISPER_MODEL", "distil-small.en")
    whisper_device: str = _get("WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = _get("WHISPER_COMPUTE_TYPE", "int8")

    # ---- TTS: Orpheus (emotional, human, inline emotion tags) ----
    tts: str = _get("TTS", "orpheus").lower()
    orpheus_model_path: str = _get(
        "ORPHEUS_MODEL_PATH",
        os.path.expanduser("~/.local/share/calliope-models/orpheus-q8.gguf"),
    )
    orpheus_voice: str = _get("ORPHEUS_VOICE", "dan")  # tara/leah/jess/leo/dan/mia/zac/zoe
    orpheus_temperature: float = float(_get("ORPHEUS_TEMPERATURE", "0.25"))  # lower = calmer
    orpheus_speed: float = float(_get("ORPHEUS_SPEED", "1.15"))  # WSOLA speed-up (pitch kept)

    # ---- Sales script ----
    company_name: str = _get("COMPANY_NAME", "Acme Solar")
    agent_name: str = _get("AGENT_NAME", "Alex")
    product_pitch: str = _get(
        "PRODUCT_PITCH",
        "residential solar panels that cut electricity bills by up to 40%",
    )

    def validate(self) -> None:
        if self.brain not in ("ollama", "gemini"):
            raise ValueError(f"BRAIN must be 'ollama' or 'gemini', got {self.brain!r}")
        if self.brain == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "BRAIN=gemini but GEMINI_API_KEY is empty. "
                "Get a free key at https://aistudio.google.com/apikey"
            )


config = Config()
