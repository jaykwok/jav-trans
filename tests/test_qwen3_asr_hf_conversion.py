from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tools.asr.convert_qwen3_asr_to_hf import convert_qwen3_asr_checkpoint_to_hf


def _write_template(template_dir: Path) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "config.json": {
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "model_type": "qwen3_asr",
            "audio_config": {"d_model": 4},
            "text_config": {"hidden_size": 4},
        },
        "generation_config.json": {"eos_token_id": [1, 2], "pad_token_id": 2},
        "processor_config.json": {"processor_class": "Qwen3ASRProcessor"},
        "tokenizer_config.json": {"model_max_length": 1024},
    }
    for name, payload in payloads.items():
        (template_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (template_dir / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (template_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")


def test_convert_qwen3_asr_non_hf_checkpoint_to_hf_layout(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {
            "thinker.audio_tower.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "thinker.model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        },
        source_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "thinker_config": {}}),
        encoding="utf-8",
    )

    template_dir = tmp_path / "template"
    _write_template(template_dir)
    output_dir = tmp_path / "output"

    report = convert_qwen3_asr_checkpoint_to_hf(
        source_model_dir=source_dir,
        output_dir=output_dir,
        template_dir=template_dir,
        max_shard_size_bytes=32,
    )

    assert report["tensor_count"] == 2
    assert report["key_rewrite"] == "thinker.* -> model.*"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "processor_config.json").exists()

    tensors = {}
    for shard in sorted(output_dir.glob("model-*.safetensors")) or [output_dir / "model.safetensors"]:
        tensors.update(load_file(shard))

    assert set(tensors) == {
        "model.audio_tower.weight",
        "model.model.embed_tokens.weight",
    }
    assert torch.equal(
        tensors["model.audio_tower.weight"],
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    )

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert index["weight_map"]["model.audio_tower.weight"].endswith(".safetensors")


def test_convert_qwen3_asr_dry_run_does_not_write_weights(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {"thinker.model.norm.weight": torch.ones(2, dtype=torch.float32)},
        source_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    template_dir = tmp_path / "template"
    _write_template(template_dir)
    output_dir = tmp_path / "output"

    report = convert_qwen3_asr_checkpoint_to_hf(
        source_model_dir=source_dir,
        output_dir=output_dir,
        template_dir=template_dir,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["tensor_count"] == 1
    assert not (output_dir / "model.safetensors").exists()
    assert not (output_dir / "config.json").exists()
