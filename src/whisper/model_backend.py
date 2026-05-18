import gc
import logging
import os
import re
import traceback
import zlib
from pathlib import Path
from typing import Any, Callable

from audio.loading import load_audio_16k_mono
from utils import hf_progress
from utils.model_paths import WHISPER_MODEL_PATH, resolve_model_spec
from whisper.generation_budget import apply_generation_budget


hf_progress.install()
logger = logging.getLogger(__name__)


WHISPER_PRESETS: dict[str, dict[str, Any]] = {
    "anime-whisper": {
        "repo_id": "litagin/anime-whisper",
        "beams": 1,
        "no_repeat_ngram": 0,
        "max_new_tokens": 444,
        "backend_label": "AnimeWhisper",
        "timestamp_mode": "forced",
        "forced_fail_ratio": 0.3,
    },
    "whisper-ja-anime-v0.3": {
        "repo_id": "efwkjn/whisper-ja-anime-v0.3",
        "beams": 5,
        "no_repeat_ngram": 5,
        "max_new_tokens": 444,
        "backend_label": "WhisperJaAnimeV03",
        "timestamp_mode": "forced",
        "forced_fail_ratio": 0.3,
    },
    "whisper-ja-1.5b": {
        "repo_id": "efwkjn/whisper-ja-1.5B",
        "beams": 5,
        "no_repeat_ngram": 5,
        "max_new_tokens": 444,
        "backend_label": "WhisperJa15B",
        "timestamp_mode": "forced",
        "forced_fail_ratio": 0.3,
    },
}


def _notify(on_stage: Callable[[str], None] | None, message: str) -> None:
    if on_stage:
        on_stage(message)


def _post_clean_whisper_text(text: str) -> str:
    cleaned = re.sub(r"\([^)]{1,10}\)", "", str(text or ""))
    cleaned = re.sub(r"【[^】]+?】", "", cleaned)
    cleaned = re.sub(r"(.)\1{6,}", lambda match: match.group(1) * 6, cleaned)
    return cleaned.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


_ASR_INITIAL_PROMPT_MAX_TOKENS = max(
    0,
    _env_int("ASR_INITIAL_PROMPT_MAX_TOKENS", 180),
)
_ASR_MIN_EFFECTIVE_NEW_TOKENS = max(
    1,
    _env_int("ASR_MIN_EFFECTIVE_NEW_TOKENS", 64),
)
_QUALITY_SIGNAL_FIELDS = (
    "avg_logprob",
    "no_speech_prob",
    "compression_ratio",
)


def _cap_initial_prompt_ids(prompt_ids):
    original_count = int(prompt_ids.numel())
    max_prompt_tokens = _asr_initial_prompt_max_tokens()
    if max_prompt_tokens <= 0 or original_count <= max_prompt_tokens:
        return prompt_ids, original_count, original_count

    capped = prompt_ids[..., -max_prompt_tokens:]
    kept_count = int(capped.numel())
    logger.warning(
        "[ASR] initial_prompt token cap: original=%d, kept=%d",
        original_count,
        kept_count,
    )
    return capped, original_count, kept_count


def _asr_initial_prompt_max_tokens() -> int:
    return max(
        0,
        _env_int("ASR_INITIAL_PROMPT_MAX_TOKENS", _ASR_INITIAL_PROMPT_MAX_TOKENS),
    )


def _asr_min_effective_new_tokens() -> int:
    return max(
        1,
        _env_int("ASR_MIN_EFFECTIVE_NEW_TOKENS", _ASR_MIN_EFFECTIVE_NEW_TOKENS),
    )


def _is_prompt_overflow_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "decoder_input_ids" in message and (
        "exceeds" in message or "max_length" in message
    )


def _generation_sequences(output: Any) -> Any:
    return getattr(output, "sequences", output)


def _generate_deterministic(
    model: Any,
    processor: Any,
    input_features: Any,
    generate_kwargs: dict[str, Any],
) -> tuple[Any, bool]:
    del processor
    kwargs = dict(generate_kwargs)
    kwargs["return_dict_in_generate"] = True
    kwargs["output_scores"] = True
    kwargs.pop("temperature", None)

    try:
        output = model.generate(input_features, do_sample=False, **kwargs)
        return output, "prompt_ids" in kwargs
    except RuntimeError as generate_exc:
        if _is_prompt_overflow_error(generate_exc):
            logger.warning("[ASR] prompt overflow; dropping this chunk")
        raise


def _no_speech_token_id(model: Any, processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            token_id = convert("<|nospeech|>")
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        except Exception:
            pass

    generation_config = getattr(model, "generation_config", None)
    token_id = getattr(generation_config, "no_speech_token_id", None)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    return None


def _extract_generation_quality_signals(
    model: Any,
    processor: Any,
    output: Any,
    raw_text: str,
) -> dict[str, float | None]:
    scores = getattr(output, "scores", None)
    try:
        has_scores = scores is not None and len(scores) > 0
    except TypeError:
        has_scores = scores is not None
    if not has_scores:
        return {field: None for field in _QUALITY_SIGNAL_FIELDS}

    return {
        "avg_logprob": _extract_avg_logprob(model, output),
        "no_speech_prob": _extract_no_speech_prob(model, processor, scores),
        "compression_ratio": _compression_ratio(raw_text),
    }


def _extract_avg_logprob(model: Any, output: Any) -> float | None:
    try:
        import torch

        transition_scores = model.compute_transition_scores(
            _generation_sequences(output),
            output.scores,
            beam_indices=getattr(output, "beam_indices", None),
            normalize_logits=True,
        )
        finite = transition_scores[torch.isfinite(transition_scores)]
        if finite.numel() == 0:
            return None
        return float(finite.mean().item())
    except Exception:
        return None


def _extract_no_speech_prob(model: Any, processor: Any, scores: Any) -> float | None:
    try:
        import torch

        token_id = _no_speech_token_id(model, processor)
        if token_id is None:
            return None
        first_step = scores[0]
        if token_id >= int(first_step.shape[-1]):
            return None
        probs = torch.softmax(first_step, dim=-1)
        return float(probs[0, token_id].item())
    except Exception:
        return None


def _compression_ratio(raw_text: str) -> float | None:
    try:
        raw_bytes = str(raw_text or "").encode("utf-8", "ignore")
        compressed_len = len(zlib.compress(raw_bytes))
        return float(compressed_len / max(len(raw_bytes), 1))
    except Exception:
        return None


def _copy_quality_signals(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    for field in _QUALITY_SIGNAL_FIELDS:
        if field in source:
            target[field] = source.get(field)
    return target


def _normalize_generation_config_for_deterministic_asr(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    if getattr(generation_config, "temperature", None) not in {None, 1.0}:
        generation_config.temperature = None

    if getattr(generation_config, "pad_token_id", None) is None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            generation_config.pad_token_id = eos_token_id


class WhisperModelBackend:
    is_subprocess = False
    accepts_contexts = False
    def __init__(
        self,
        *,
        preset_name: str,
        repo_id: str,
        model_path: str,
        generation_kwargs: dict[str, Any],
        post_clean_fn: Callable[[str], str] | None = None,
        unload_every: int = 200,
        backend_label: str = "Whisper",
        timestamp_mode: str = "forced",
        forced_fail_ratio: float = 0.3,
    ):
        from whisper.local_backend import ALIGNER_BATCH_SIZE

        self.preset_name = preset_name
        self.repo_id = repo_id
        self.model_path = model_path
        self.generation_kwargs = dict(generation_kwargs)
        self.post_clean_fn = post_clean_fn or (lambda text: str(text or "").strip())
        self.unload_every = unload_every
        self.backend_label = backend_label
        self.timestamp_mode = timestamp_mode
        self.forced_fail_ratio = forced_fail_ratio
        self.request_batch_size = 1
        self.align_batch_size = ALIGNER_BATCH_SIZE
        self._model = None
        self._processor = None
        self._align_backend = None
        self._warned_contexts = False

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._model is not None and self._processor is not None:
            return

        self.model_path = resolve_model_spec(
            WHISPER_MODEL_PATH or None,
            self.repo_id,
            download=True,
        )
        _notify(on_stage, f"加载 {self.backend_label} 模型...")
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except Exception as exc:
            detail = "".join(traceback.format_exception(exc))
            raise RuntimeError(
                "Failed to import transformers Whisper classes:\n" + detail
            ) from exc
        import torch

        self._processor = WhisperProcessor.from_pretrained(self.model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="cuda",
        )
        _normalize_generation_config_for_deterministic_asr(self._model)

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._model is None and self._processor is None:
            return

        _notify(on_stage, f"卸载 {self.backend_label} 模型...")
        try:
            del self._model
        except Exception:
            pass
        try:
            del self._processor
        except Exception:
            pass
        self._model = None
        self._processor = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_forced_aligner(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._align_backend is not None:
            self._align_backend.unload_forced_aligner(on_stage=on_stage)

    def close(self) -> None:
        self.unload_model()
        self.unload_forced_aligner()
        if self._align_backend is not None:
            self._align_backend.close()
            self._align_backend = None

    def _ensure_align_backend(self):
        if self._align_backend is None:
            from whisper.local_backend import LocalAsrBackend

            self._align_backend = LocalAsrBackend("cuda")
        return self._align_backend

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        if contexts and any(contexts):
            if not getattr(self, "_warned_contexts", False):
                _notify(on_stage, f"[WARN] {self.preset_name} ignores contexts")
                self._warned_contexts = True
        if initial_prompts is not None and len(initial_prompts) != len(audio_paths):
            raise ValueError(
                f"initial_prompt count mismatch: audio_paths={len(audio_paths)}, initial_prompts={len(initial_prompts)}"
            )

        self.load(on_stage=on_stage)

        import torch
        from whisper.local_backend import _clean_master_text, _get_wav_duration

        results = []
        _notify(on_stage, f"{self.backend_label} 文本生成开始，共 {len(audio_paths)} 条...")

        for idx, audio_path in enumerate(audio_paths):
            normalized_path = str(Path(audio_path).resolve())
            duration = _get_wav_duration(normalized_path)
            budget = None
            initial_prompt = (
                str(initial_prompts[idx] or "").strip()
                if initial_prompts is not None
                else ""
            )

            try:
                audio, _sample_rate = load_audio_16k_mono(normalized_path)
                input_features = self._processor(
                    audio,
                    return_tensors="pt",
                    sampling_rate=16000,
                ).input_features.to("cuda")
                input_features = input_features.to(torch.float16)

                with torch.inference_mode():
                    generate_kwargs = dict(self.generation_kwargs)
                    generate_kwargs["max_length"] = None  # override model generation_config to avoid conflict with max_new_tokens
                    generate_kwargs["forced_decoder_ids"] = (
                        self._processor.get_decoder_prompt_ids(
                            language="ja",
                            task="transcribe",
                        )
                    )
                    prompt_token_count = 0
                    prompt_warning = ""
                    prompt_ids = None
                    if initial_prompt:
                        try:
                            prompt_ids = self._processor.get_prompt_ids(
                                initial_prompt,
                                return_tensors="pt",
                            ).to("cuda")
                            prompt_ids, original_count, prompt_token_count = (
                                _cap_initial_prompt_ids(prompt_ids)
                            )
                            if original_count != prompt_token_count:
                                prompt_warning = (
                                    f"{self.backend_label} initial_prompt token cap: "
                                    f"original={original_count}, kept={prompt_token_count}"
                                )
                        except Exception as prompt_exc:
                            prompt_warning = (
                                f"{self.backend_label} initial_prompt skipped: {prompt_exc}"
                            )
                    generate_kwargs, prompt_ids, budget = apply_generation_budget(
                        model=self._model,
                        generate_kwargs=generate_kwargs,
                        prompt_ids=prompt_ids,
                        min_effective_new_tokens=_asr_min_effective_new_tokens(),
                    )
                    prompt_token_count = int(budget.prompt_tokens_kept)
                    if budget.clipped_prompt_tokens:
                        prompt_budget_warning = (
                            f"{self.backend_label} generation budget clipped initial_prompt: "
                            f"original={budget.prompt_tokens_original}, kept={budget.prompt_tokens_kept}"
                        )
                        prompt_warning = (
                            f"{prompt_warning}; {prompt_budget_warning}"
                            if prompt_warning
                            else prompt_budget_warning
                        )
                    if budget.clipped_max_new_tokens:
                        token_budget_warning = (
                            f"{self.backend_label} generation budget clipped max_new_tokens: "
                            f"configured={budget.configured_max_new_tokens}, "
                            f"effective={budget.effective_max_new_tokens}, "
                            f"limit={budget.model_max_target_positions}"
                        )
                        prompt_warning = (
                            f"{prompt_warning}; {token_budget_warning}"
                            if prompt_warning
                            else token_budget_warning
                        )
                    output, _used_prompt_ids = _generate_deterministic(
                        self._model,
                        self._processor,
                        input_features,
                        generate_kwargs,
                    )
                    predicted_ids = _generation_sequences(output)
                    text = self._processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True,
                    )[0]
                    text = self.post_clean_fn(text)
                    quality_signals = _extract_generation_quality_signals(
                        self._model,
                        self._processor,
                        output,
                        text,
                    )

                master_text = _clean_master_text(text)

                results.append(
                    {
                        "text": master_text,
                        "raw_text": text,
                        "duration": duration,
                        "language": "Japanese",
                        "normalized_path": normalized_path,
                        **quality_signals,
                        "asr_generation": {
                            "backend": self.preset_name,
                            "budget": budget.as_dict() if budget else {},
                            "error_kind": None,
                            "error_detail": "",
                        },
                        "log": [
                            f"{self.backend_label} ASR 输出模式: text_only",
                            (
                                f"{self.backend_label} max_new_tokens: "
                                f"{budget.effective_max_new_tokens}/{budget.configured_max_new_tokens}"
                                if budget
                                else ""
                            ),
                            *(
                                [
                                    f"{self.backend_label} initial_prompt tokens: {prompt_token_count}"
                                ]
                                if prompt_token_count
                                else []
                            ),
                            *([prompt_warning] if prompt_warning else []),
                        ],
                    }
                )
            except Exception as e:
                _notify(on_stage, f"[ERROR] {self.backend_label} 转录异常 {audio_path}: {e}")
                error_kind = (
                    "overflow"
                    if isinstance(e, RuntimeError) and _is_prompt_overflow_error(e)
                    else "generation_error"
                )
                results.append(
                    {
                        "text": "",
                        "raw_text": "",
                        "duration": duration,
                        "language": "Japanese",
                        "normalized_path": normalized_path,
                        "avg_logprob": None,
                        "no_speech_prob": None,
                        "compression_ratio": None,
                        "asr_generation": {
                            "backend": self.preset_name,
                            "budget": budget.as_dict() if budget else {},
                            "error_kind": error_kind,
                            "error_detail": str(e),
                        },
                        "log": [f"{self.backend_label} 转录异常: {e}"],
                    }
                )

            if (idx + 1) % self.unload_every == 0:
                torch.cuda.empty_cache()

        return results

    def finalize_text_results(
        self,
        text_results: list[dict[str, Any]],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict[str, Any], list[str]]]:
        if not text_results:
            return []

        from whisper.local_backend import align_text_to_words, normalize_word_dicts
        from whisper.timestamp_fallback import build_word_timestamps_fallback

        mode = self.timestamp_mode
        aligner_handle = None

        if mode == "forced":
            aligner_handle = self._ensure_align_backend()._ensure_forced_aligner(on_stage=on_stage)

        results_forced = []
        forced_fail_count = 0
        total_valid_count = 0

        for res in text_results:
            master_text = res.get("text", "")
            normalized_path = res.get("normalized_path", "")
            duration = res.get("duration", 0.0)
            raw_text = res.get("raw_text", "")
            language = res.get("language", "Japanese")

            if not master_text:
                results_forced.append(None)
                continue

            total_valid_count += 1
            if mode == "forced":
                try:
                    word_dicts, alignment_mode = align_text_to_words(
                        normalized_path,
                        raw_text or master_text,
                        language,
                        aligner_handle=aligner_handle,
                    )

                    if not word_dicts:
                        forced_fail_count += 1
                        results_forced.append(None)
                    else:
                        results_forced.append(
                            _copy_quality_signals(
                                res,
                                {
                                "words": word_dicts,
                                "text": master_text,
                                "raw_text": raw_text,
                                "alignment_mode": "forced_aligner",
                                "duration": duration,
                                "language": language,
                                },
                            )
                        )
                except Exception:
                    forced_fail_count += 1
                    results_forced.append(None)
            else:
                results_forced.append(None)

        if mode == "forced" and total_valid_count > 0:
            fail_ratio = forced_fail_count / total_valid_count
            if fail_ratio > self.forced_fail_ratio:
                _notify(
                    on_stage,
                    f"[WARN] {self.backend_label} forced_align 失败率 {fail_ratio:.1%} > 阈值 {self.forced_fail_ratio:.1%}，整批降级 vad_ratio",
                )
                mode = "vad_ratio"

        final_results = []
        for i, res in enumerate(text_results):
            master_text = res.get("text", "")
            normalized_path = res.get("normalized_path", "")
            duration = res.get("duration", 0.0)
            raw_text = res.get("raw_text", "")
            language = res.get("language", "Japanese")
            log = res.get("log", []).copy()

            if not master_text:
                final_results.append(
                    (
                        _copy_quality_signals(
                            res,
                            {
                            "words": [],
                            "text": "",
                            "raw_text": raw_text,
                            "alignment_mode": "empty",
                            "duration": duration,
                            "language": language,
                            },
                        ),
                        log,
                    )
                )
                continue

            if mode == "forced" and results_forced[i] is not None:
                log.append(f"{self.backend_label} 对齐模式: forced_aligner")
                final_results.append((results_forced[i], log))
            else:
                word_dicts, alignment_mode, fallback_meta = build_word_timestamps_fallback(
                    master_text,
                    0.0,
                    duration,
                    audio_path=normalized_path,
                )
                normalized = normalize_word_dicts(word_dicts)
                log.append(f"{self.backend_label} 对齐模式: {alignment_mode}")
                if fallback_meta:
                    if fallback_meta.get("speech_span_count", 0):
                        log.append(
                            f"{self.backend_label} VAD 回退语音区间: {fallback_meta['speech_span_count']}"
                        )
                    elif fallback_meta.get("vad_error"):
                        log.append(f"{self.backend_label} VAD 回退异常: {fallback_meta['vad_error']}")

                final_results.append(
                    (
                        _copy_quality_signals(
                            res,
                            {
                            "words": normalized,
                            "text": master_text,
                            "raw_text": raw_text,
                            "alignment_mode": alignment_mode,
                            "duration": duration,
                            "language": language,
                            },
                        ),
                        log,
                    )
                )

        return final_results


def create_whisper_model_backend(preset_name: str, device: str) -> WhisperModelBackend:
    del device
    preset = WHISPER_PRESETS[preset_name]
    repo_id = str(preset["repo_id"])
    model_path = resolve_model_spec(
        WHISPER_MODEL_PATH or None,
        repo_id,
    )
    return WhisperModelBackend(
        preset_name=preset_name,
        repo_id=repo_id,
        model_path=model_path,
        generation_kwargs={
            "num_beams": _env_int("WHISPER_BEAMS", int(preset["beams"])),
            "no_repeat_ngram_size": _env_int(
                "WHISPER_NO_REPEAT_NGRAM",
                int(preset["no_repeat_ngram"]),
            ),
            "max_new_tokens": _env_int(
                "WHISPER_MAX_NEW_TOKENS",
                int(preset["max_new_tokens"]),
            ),
        },
        post_clean_fn=_post_clean_whisper_text,
        unload_every=_env_int("WHISPER_UNLOAD_EVERY", 200),
        backend_label=str(preset["backend_label"]),
        timestamp_mode=os.getenv(
            "WHISPER_TIMESTAMP_MODE",
            str(preset["timestamp_mode"]),
        ).strip().lower(),
        forced_fail_ratio=_env_float(
            "WHISPER_FORCED_FAIL_RATIO",
            float(preset["forced_fail_ratio"]),
        ),
    )
