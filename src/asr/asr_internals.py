"""CueQC Mamba v4 binary ASR internals capture via teacher-forced forward.

``AsrInternalsCapturer`` extracts, for one ASR candidate, the two signal sources
CueQC v4 binary needs values that the public ``qwen_asr.transcribe()`` API does not return:

* ``asr_frames``  — Qwen3-ASR encoder (audio-tower) hidden states for the chunk
* token-level logprob / entropy / top1-top2 margin — via a single
  teacher-forced ``model(...)`` forward over ``prompt + generated_text_ids``

Both training (offline) and runtime reuse this class so the feature pipeline is
identical end-to-end. The capturer holds no model of its own when a wrapper is
passed in — at runtime it reuses the already-loaded ``LocalAsrBackend.model``
to avoid doubling VRAM.

The teacher-forced logits reproduce the same positions ASR decoding produced
because the wrapper is greedy (default ``generate``). If the ASR backend ever
switches to beam/sampling, this becomes a "re-score under greedy" path rather
than a strict equivalence — re-evaluate then.
"""
from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any

import numpy as np

# Torchcodec ships a broken DLL on this Windows env; stub it before torch loads
# (mirrors cueqc_model.py / extract_features.py).
if "torchcodec" not in sys.modules:
    for _mod_name in ("torchcodec", "torchcodec.decoders"):
        _stub = types.ModuleType(_mod_name)
        _stub.__spec__ = ModuleSpec(_mod_name, loader=None)
        _stub.__path__ = []
        sys.modules[_mod_name] = _stub

SAMPLE_RATE = 16000


def _qwen3_asr_audio_output_lengths(input_lengths):
    """Per-audio output frame length for the Qwen3-ASR audio tower.

    Mirrors ``src/boundary/ja/features.py`` (originally from the qwen_asr
    processor). Used to slice the flattened encoder hidden state back into
    per-chunk sequences.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class AsrInternalsCapturer:
    """Capture ASR encoder features + per-token logits for one chunk.

    Construct with either:

    * ``wrapper=`` — a loaded ``qwen_asr.Qwen3ASRModel`` (runtime path, reuses
      ``LocalAsrBackend.model`` to avoid a second model load).
    * ``model_spec=`` — a repo id / local path to load its own wrapper (offline
      feature extraction path).
    """

    def __init__(
        self,
        *,
        wrapper: Any | None = None,
        model_spec: str | None = None,
        device: str = "auto",
        dtype: str | None = None,
        context: str | None = None,
        force_language: str | None = None,
    ) -> None:
        import torch  # noqa: F401  (ensure import ordering)

        if wrapper is None and model_spec is None:
            raise ValueError("AsrInternalsCapturer requires wrapper= or model_spec=")

        if wrapper is None:
            wrapper = self._load_wrapper(model_spec, device=device, dtype=dtype)

        self.wrapper = wrapper
        self.model = wrapper.model  # Qwen3ASRForConditionalGeneration (top-level: forward is a stub)
        # The actual generative LM is the thinker; teacher-forced logits come
        # from thinker.forward() (returns Qwen3ASRThinkerCausalLMOutputWithPast
        # with .logits [B, T, vocab]). The top-level .forward() is unimplemented.
        self.thinker = self.model.thinker
        self.processor = wrapper.processor
        self.device = getattr(wrapper, "device", None) or self._infer_device(self.model)
        self.dtype = getattr(wrapper, "dtype", None) or self._infer_dtype(self.model)

        # Prompt configuration. Defaults follow LocalAsrBackend: language forced
        # to ASR_LANGUAGE (Japanese) and empty context unless overridden.
        self.context = context if context is not None else os.getenv("ASR_CONTEXT", "").strip()
        if force_language is not None:
            self.force_language = force_language or None
        else:
            from asr.local_backend import _asr_force_language, _asr_language

            self.force_language = _asr_language() if _asr_force_language() else None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_wrapper(model_spec: str, *, device: str, dtype: str | None):
        from qwen_asr import Qwen3ASRModel
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs: dict[str, Any] = {
            "dtype": AsrInternalsCapturer._resolve_dtype(dtype, device),
            "device_map": device,
            "max_inference_batch_size": 1,
            "max_new_tokens": 1,
        }
        return Qwen3ASRModel.from_pretrained(model_spec, **model_kwargs)

    @staticmethod
    def _resolve_dtype(dtype: str | None, device: str):
        import torch

        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return torch.float16 if device.startswith("cuda") else torch.float32

    @staticmethod
    def _infer_device(model):
        import torch

        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _infer_dtype(model):
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return None

    # ------------------------------------------------------------------ audio
    def _load_audio_slice(self, wav_path_or_array, start_s: float, end_s: float) -> np.ndarray:
        """Return mono 16kHz float32 audio for [start_s, end_s]."""
        if isinstance(wav_path_or_array, np.ndarray):
            audio = np.asarray(wav_path_or_array, dtype=np.float32).reshape(-1)
        else:
            import soundfile as sf

            audio, sr = sf.read(str(wav_path_or_array), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import scipy.signal

                audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        s = max(0, int(round(start_s * SAMPLE_RATE)))
        e = min(audio.shape[0], max(s + 1, int(round(end_s * SAMPLE_RATE))))
        return np.ascontiguousarray(audio[s:e], dtype=np.float32)

    # --------------------------------------------------------------- prompt
    def _build_prompt(self) -> str:
        """Single-sourced prompt builder: reuse the wrapper's own method."""
        return self.wrapper._build_text_prompt(
            context=self.context or "",
            force_language=self.force_language,
        )

    def _tokenize_generated(self, text: str) -> list[int]:
        """Tokenize the ASR-generated text (no special tokens) for teacher forcing.

        The generated region sits after the prompt + ``language X<asr_text>`` tail.
        We encode the raw text the same way the decoder would have produced it.
        """
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(i) for i in ids]

    # ---------------------------------------------------------------- extract
    def extract(
        self,
        wav_path_or_array,
        text: str,
        *,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        """Capture ASR internals for one chunk.

        Returns ``asr_frames`` [T, D_asr], the generated ``token_ids`` [L], and
        per-token ``token_logprobs`` / ``token_entropies`` / ``token_top1_top2_margins``
        [L]. Raises on any failure; callers fall back to keep (v3 §2.4).
        """
        import torch

        audio = self._load_audio_slice(wav_path_or_array, start_s, end_s if end_s is not None else 0.0)
        if audio.shape[0] < SAMPLE_RATE // 20:  # < 50ms -> pad to avoid degenerate encoder input
            audio = np.pad(audio, (0, SAMPLE_RATE // 2 - audio.shape[0]))

        prompt_str = self._build_prompt()
        gen_ids = self._tokenize_generated(str(text or ""))
        if not gen_ids:
            gen_ids = []  # empty-text candidate; token trace will be a zero row downstream

        # Build full input: processor encodes prompt + audio, then we splice the
        # generated text ids onto the prompt ids so teacher-forced logits cover
        # every generated position.
        inputs = self.processor(
            text=[prompt_str],
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )
        prompt_input_ids = inputs["input_ids"][0].tolist()  # [P]
        prompt_len = len(prompt_input_ids)

        full_ids = prompt_input_ids + gen_ids
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        # attention_mask: all ones (real prompt + audio placeholders + generated ids)
        attention_mask = torch.ones_like(full_input_ids)

        # Move audio inputs to model device/dtype.
        feature_attention_mask = inputs["feature_attention_mask"].to(self.device)
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            audio_features = self.thinker.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )  # [B, T_audio_raw, D_asr]; for batch=1 some versions return [T, D]
            # Normalize to [B, T, D] then slice to the real (unpadded) length.
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)
            input_lengths = feature_attention_mask.sum(dim=1)
            out_len = int(_qwen3_asr_audio_output_lengths(input_lengths)[0].item())
            asr_frames = audio_features[0, :out_len].detach().float().cpu().numpy()

            # Teacher-forced forward over prompt + generated ids, on the thinker
            # (the top-level model.forward is an unimplemented stub).
            outputs = self.thinker(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                use_cache=False,
            )
            logits = outputs.logits  # [1, P+L, V]

        asr_frames = np.ascontiguousarray(asr_frames, dtype=np.float32)

        if not gen_ids:
            return {
                "asr_frames": asr_frames,
                "token_ids": np.zeros((0,), dtype=np.int64),
                "token_logprobs": np.zeros((0,), dtype=np.float32),
                "token_entropies": np.zeros((0,), dtype=np.float32),
                "token_top1_top2_margins": np.zeros((0,), dtype=np.float32),
                "decoded_tokens": [],
                "has_timestamps": False,
            }

        L = len(gen_ids)
        # Causal shift: logits at position i predict token i+1. The generated
        # token at sequence position (prompt_len + k) is predicted by logits at
        # position (prompt_len - 1 + k).
        gen_logits = logits[0, prompt_len - 1 : prompt_len - 1 + L, :]  # [L, V]
        assert gen_logits.shape[0] == L, f"logits length {gen_logits.shape[0]} != L {L}"

        log_probs = torch.log_softmax(gen_logits.float(), dim=-1)
        target_ids = torch.tensor(gen_ids, dtype=torch.long, device=log_probs.device)
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        top2 = torch.topk(log_probs, k=2, dim=-1).values  # [L, 2]
        margin = top2[:, 0] - top2[:, 1]

        decoded_tokens = [self.processor.tokenizer.decode([int(i)]) for i in gen_ids]

        return {
            "asr_frames": asr_frames,
            "token_ids": np.asarray(gen_ids, dtype=np.int64),
            "token_logprobs": token_logprobs.detach().float().cpu().numpy().astype(np.float32),
            "token_entropies": entropy.detach().float().cpu().numpy().astype(np.float32),
            "token_top1_top2_margins": margin.detach().float().cpu().numpy().astype(np.float32),
            "decoded_tokens": decoded_tokens,
            "has_timestamps": False,
        }

    def close(self) -> None:
        """Release the wrapper only if this capturer owns it (offline path)."""
        import torch

        # Runtime path shares the wrapper with LocalAsrBackend; do not free it.
        # Offline path loaded its own; let GC handle it. Empty CUDA cache.
        if self.device is not None and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


def extract_candidate_internals(
    capturer: AsrInternalsCapturer,
    *,
    audio_path: str,
    text: str,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Thin convenience wrapper used by both extract tools and the refiner."""
    return capturer.extract(audio_path, text, start_s=start_s, end_s=end_s)
