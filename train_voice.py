"""Fine-tune Orpheus on ONE custom voice (LoRA), then export a GGUF the Studio can use.

Run by the Studio (in the ISOLATED .venv-train):  python train_voice.py <voice_name>
Reads   voice_dataset/voices/<name>/clips/*.wav  +  metadata.csv
Writes  voice_dataset/voices/<name>/model.gguf   and updates status.json

Recipe follows Unsloth's Orpheus notebook: SNAC audio tokens at base 128266 (+layer*4096),
7 tokens/frame, wrapped in Orpheus's speech special tokens.
"""
import csv
import json
import os
import sys
import traceback

# HuggingFace's "xet" transfer backend stalls on large files here — force plain HTTP.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))


def voice_dir(name):
    return os.path.join(ROOT, "voice_dataset", "voices", name)


def set_status(d, **kw):
    p = os.path.join(d, "status.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw)
    json.dump(st, open(p, "w"))


# Orpheus special tokens
START_OF_HUMAN = 128003
END_OF_TEXT = 128009
END_OF_HUMAN = 128004
START_OF_AI = 128005
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
END_OF_AI = 128006
AUDIO_BASE = 128266
MAX_SEQ = 4096


def main(name):
    d = voice_dir(name)
    if not os.path.isdir(d):
        raise SystemExit(f"no such voice: {name}")
    print(f"[train] voice={name}", flush=True)

    import numpy as np
    import torch
    import soundfile as sf
    from snac import SNAC
    from unsloth import FastLanguageModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}", flush=True)

    # ---- 1) load base model + tokenizer ----
    print("[train] loading base Orpheus (unsloth/orpheus-3b-0.1-ft)…", flush=True)
    # 4-bit (QLoRA) keeps training VRAM low so it can coexist with the Studio.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/orpheus-3b-0.1-ft",
        max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # ---- 2) SNAC encoder ----
    print("[train] loading SNAC…", flush=True)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    def audio_to_codes(wav_path):
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pk = float(np.abs(audio).max())          # normalize loudness for consistent training
        if pk > 1e-4:
            audio = audio / pk * 0.95
        t = torch.tensor(audio, dtype=torch.float32, device=device).view(1, 1, -1)
        with torch.inference_mode():
            codes = snac.encode(t)
        out = []
        n = codes[0].shape[1]
        for i in range(n):
            out.append(codes[0][0][i].item() + AUDIO_BASE)
            out.append(codes[1][0][2 * i].item() + AUDIO_BASE + 4096)
            out.append(codes[2][0][4 * i].item() + AUDIO_BASE + 2 * 4096)
            out.append(codes[2][0][4 * i + 1].item() + AUDIO_BASE + 3 * 4096)
            out.append(codes[1][0][2 * i + 1].item() + AUDIO_BASE + 4 * 4096)
            out.append(codes[2][0][4 * i + 2].item() + AUDIO_BASE + 5 * 4096)
            out.append(codes[2][0][4 * i + 3].item() + AUDIO_BASE + 6 * 4096)
        return out

    # ---- 3) build dataset ----
    rows = list(csv.reader(open(os.path.join(d, "metadata.csv"))))
    print(f"[train] {len(rows)} clips", flush=True)
    examples = []
    for rel, text in rows:
        wav = os.path.join(d, rel)
        if not os.path.exists(wav):
            continue
        codes = audio_to_codes(wav)
        text_ids = tokenizer(f"{name}: {text}", add_special_tokens=False).input_ids
        ids = ([START_OF_HUMAN] + text_ids + [END_OF_TEXT, END_OF_HUMAN,
               START_OF_AI, START_OF_SPEECH] + codes + [END_OF_SPEECH, END_OF_AI])
        if len(ids) > MAX_SEQ:
            continue
        examples.append({"input_ids": ids, "labels": list(ids),
                         "attention_mask": [1] * len(ids)})
    print(f"[train] usable examples: {len(examples)}", flush=True)
    if len(examples) < 10:
        raise SystemExit("not enough usable clips (need ~10+)")

    from datasets import Dataset
    dataset = Dataset.from_list(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 128263

    def collate(feats):
        m = max(len(f["input_ids"]) for f in feats)
        ii, lb, am = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            ii.append(f["input_ids"] + [pad_id] * k)
            lb.append(f["labels"] + [-100] * k)
            am.append(f["attention_mask"] + [0] * k)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(lb),
                "attention_mask": torch.tensor(am)}

    # ---- 4) train ----
    from transformers import Trainer, TrainingArguments
    steps_per_epoch = max(1, len(examples) // 4)
    max_steps = int(min(500, max(120, steps_per_epoch * 10)))
    print(f"[train] max_steps={max_steps}", flush=True)
    FastLanguageModel.for_training(model)
    trainer = Trainer(
        model=model, train_dataset=dataset, data_collator=collate,
        args=TrainingArguments(
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            warmup_steps=5, max_steps=max_steps, learning_rate=2e-4,
            logging_steps=5, optim="adamw_8bit", weight_decay=0.001,
            lr_scheduler_type="linear", seed=3407, output_dir=os.path.join(d, "outputs"),
            report_to="none",
        ),
    )
    trainer.train()

    # Save the LoRA adapter immediately (safety: export below can be slow/flaky).
    lora_dir = os.path.join(d, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print("[train] LoRA adapter saved", flush=True)

    # ---- 5) export merged GGUF for llama.cpp ----
    print("[train] exporting GGUF (q8_0)… this builds llama.cpp on first run", flush=True)
    import glob
    gguf_dir = os.path.join(d, "gguf_export")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q8_0")
    # Unsloth may write to gguf_export_gguf/ (appends _gguf) — search recursively.
    cands = [c for c in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True)
             if os.path.basename(c) != "model.gguf"]
    if not cands:
        raise SystemExit("GGUF export produced no file")
    found = max(cands, key=os.path.getmtime)
    final = os.path.join(d, "model.gguf")
    os.replace(found, final)
    set_status(d, state="ready", gguf=final)
    print(f"[train] DONE -> {final}", flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    d = voice_dir(name)
    try:
        main(name)
    except Exception:
        traceback.print_exc()
        set_status(d, state="failed")
        sys.exit(1)
