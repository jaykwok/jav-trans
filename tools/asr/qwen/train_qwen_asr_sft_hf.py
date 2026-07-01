#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.qwen_native import build_transcription_prompt  # noqa: E402


class JsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_audio_16k(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


class Qwen3AsrNativeCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.masked_token_ids = {
            processor.tokenizer.pad_token_id,
            processor.audio_token_id,
            processor.audio_bos_token_id,
            processor.audio_eos_token_id,
        }

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios = [load_audio_16k(str(row["audio"])) for row in rows]
        prompts = [
            build_transcription_prompt(
                self.processor,
                language=str(row.get("language") or "Japanese"),
            )
            for row in rows
        ]
        texts = [
            prompt + str(row.get("text") or "") + "<|im_end|>\n"
            for prompt, row in zip(prompts, rows, strict=True)
        ]
        inputs = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = self.processor.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        labels = inputs["input_ids"].clone()
        for row_index in range(labels.shape[0]):
            response_start = int(prompt_inputs["attention_mask"][row_index].sum().item())
            labels[row_index, :response_start] = -100
        labels[inputs["attention_mask"] == 0] = -100
        for token_id in self.masked_token_ids:
            if token_id is not None:
                labels[labels == token_id] = -100
        inputs["labels"] = labels
        return dict(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Transformers Qwen3-ASR SFT")
    parser.add_argument("--model_path", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--persistent_workers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--resume_from")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments

    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForMultimodalLM.from_pretrained(args.model_path, dtype=dtype)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = JsonlDataset(args.train_file)
    eval_dataset = JsonlDataset(args.eval_file) if args.eval_file else None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory),
        dataloader_persistent_workers=bool(args.persistent_workers and args.num_workers),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers else None,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3AsrNativeCollator(processor),
    )
    trainer.train(resume_from_checkpoint=args.resume_from or None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
