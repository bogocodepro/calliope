"""In-process Orpheus TTS: llama-cpp-python (GGUF on GPU) -> SNAC 24 kHz decode.

Orpheus is the emotional, human-sounding voice (supports <laugh>, <sigh>, <chuckle>,
<gasp>, <groan>, <yawn>, <cough>, <sniffle> inline). This runs the 3B GGUF locally on
the GPU and decodes its audio tokens with SNAC — no external server.

The prompt format, the token->id math, and the SNAC frame assembly are ported verbatim
from the tested Orpheus-FastAPI project (tts_engine/inference.py + speechpipe.py) so the
audio is correct. Generation is streamed and decoded per 7-token frame for low latency.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
from snac import SNAC

CUSTOM_TOKEN_PREFIX = "<custom_token_"
ENGLISH_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"


def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Parse a <custom_token_N> string into a SNAC code id (ported from Orpheus)."""
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
    token_string = token_string.strip()
    last = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    if last == -1:
        return None
    last_token = token_string[last:]
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
    try:
        number_str = last_token[14:-1]
        return int(number_str) - 10 - ((index % 7) * 4096)
    except (ValueError, IndexError):
        return None


class OrpheusTTS:
    def __init__(self, model_path: str, voice: str = DEFAULT_VOICE,
                 device: str = "cuda", n_gpu_layers: int = -1, n_ctx: int = 4096,
                 temperature: float = 0.4, top_p: float = 0.9, lora_path=None,
                 snac_model=None):
        self.voice = voice  # allow custom speaker names (from fine-tuned voices)
        # Lower temperature => calmer, more even, less "performed" delivery.
        self.temperature = temperature
        self.top_p = top_p
        self.sr = 24000
        self.device = device if torch.cuda.is_available() else "cpu"

        # SNAC can be shared across engines to save VRAM.
        if snac_model is not None:
            self.snac = snac_model
        else:
            logger.info("Loading SNAC 24kHz decoder…")
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        from llama_cpp import Llama

        logger.info(f"Loading Orpheus GGUF on GPU ({model_path})"
                    + (f" + LoRA {lora_path}" if lora_path else "") + "…")
        kw = dict(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
        if lora_path:
            kw["lora_path"] = lora_path
        self.llm = Llama(**kw)
        logger.info(f"Orpheus ready (voice={self.voice})")

    # ---- SNAC decode (ported verbatim from Orpheus-FastAPI speechpipe.py) ----
    def _convert_to_audio(self, multiframe: List[int], count: int) -> Optional[bytes]:
        if len(multiframe) < 7:
            return None
        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]
        dev = self.device

        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=dev)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=dev)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=dev)
        ft = torch.tensor(frame, dtype=torch.int32, device=dev)

        for j in range(num_frames):
            idx = j * 7
            codes_0[j] = ft[idx]
            codes_1[j * 2] = ft[idx + 1]
            codes_1[j * 2 + 1] = ft[idx + 4]
            codes_2[j * 4] = ft[idx + 2]
            codes_2[j * 4 + 1] = ft[idx + 3]
            codes_2[j * 4 + 2] = ft[idx + 5]
            codes_2[j * 4 + 3] = ft[idx + 6]

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # Codebook indices are 0..4095. A stray code (e.g. 4096) is an out-of-range
        # index that triggers an UNRECOVERABLE CUDA assertion and kills the process.
        # Clamp to the valid range instead.
        for cb in codes:
            cb.clamp_(0, 4095)

        with torch.inference_mode():
            audio_hat = self.snac.decode(codes)
            audio_slice = audio_hat[:, :, 2048:4096]
            audio_int16 = (audio_slice * 32767).to(torch.int16)
            return audio_int16.cpu().numpy().tobytes()

    def synth_full(self, text: str) -> np.ndarray:
        """Generate ALL audio tokens, then decode them in ONE SNAC pass -> smooth
        float32 (24 kHz), no inter-chunk seams/crackle. Returns [] if nothing valid."""
        tokens: List[int] = []
        count = 0
        for tok_str in self._token_strings(text):
            tid = turn_token_into_id(tok_str, count)
            if tid is None or tid <= 0:
                continue
            tokens.append(tid)
            count += 1

        n = (len(tokens) // 7) * 7
        if n < 7:
            return np.zeros(0, dtype=np.float32)
        frame = tokens[:n]
        num_frames = n // 7
        dev = self.device

        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=dev)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=dev)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=dev)
        ft = torch.tensor(frame, dtype=torch.int32, device=dev)
        for j in range(num_frames):
            idx = j * 7
            codes_0[j] = ft[idx]
            codes_1[j * 2] = ft[idx + 1]
            codes_1[j * 2 + 1] = ft[idx + 4]
            codes_2[j * 4] = ft[idx + 2]
            codes_2[j * 4 + 1] = ft[idx + 3]
            codes_2[j * 4 + 2] = ft[idx + 5]
            codes_2[j * 4 + 3] = ft[idx + 6]

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        for cb in codes:
            cb.clamp_(0, 4095)  # valid indices 0..4095; 4096 => CUDA assert => crash
        with torch.inference_mode():
            audio_hat = self.snac.decode(codes)          # full, seamless waveform
        audio = audio_hat[0, 0].detach().cpu().numpy().astype(np.float32)
        return np.clip(audio, -1.0, 1.0)                 # guard int-overflow crackle

    def _format_prompt(self, text: str) -> str:
        return f"<|audio|>{self.voice}: {text}<|eot_id|>"

    def _token_strings(self, text: str) -> Iterator[str]:
        """Stream raw token text from llama.cpp, split on '>' like Orpheus expects."""
        stream = self.llm.create_completion(
            self._format_prompt(text),
            max_tokens=2000,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=1.1,
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if not piece:
                continue
            for part in piece.split(">"):
                yield part + ">"

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks (24 kHz) as they're generated (low latency)."""
        buffer: List[int] = []
        count = 0
        first_done = False

        def emit(window: List[int]) -> Optional[np.ndarray]:
            pcm = self._convert_to_audio(window, count)
            if pcm is None:
                return None
            return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0

        for tok_str in self._token_strings(text):
            tid = turn_token_into_id(tok_str, count)
            if tid is None or tid <= 0:
                continue
            buffer.append(tid)
            count += 1
            if not first_done:
                if count >= 7:
                    a = emit(buffer[-7:])
                    if a is not None:
                        first_done = True
                        yield a
            elif count % 7 == 0:
                if len(buffer) >= 49:
                    w = buffer[-49:]
                elif len(buffer) >= 28:
                    w = buffer[-28:]
                else:
                    continue
                a = emit(w)
                if a is not None:
                    yield a

        # flush the tail
        if len(buffer) >= 49:
            w = buffer[-49:]
        elif len(buffer) >= 28:
            w = buffer[-28:]
        elif len(buffer) >= 7:
            pad = [buffer[-1]] * (28 - len(buffer))
            w = buffer + pad
        else:
            return
        a = emit(w)
        if a is not None:
            yield a
