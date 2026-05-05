import gc
import os
import re
from pathlib import Path
from typing import Any, Callable

from audio.loading import load_audio_16k_mono
from utils import hf_progress
from utils.model_paths import WHISPER_MODEL_PATH, resolve_model_spec


hf_progress.install()


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
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch

        self._processor = WhisperProcessor.from_pretrained(self.model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="cuda",
        )

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
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        if contexts and any(contexts):
            if not getattr(self, "_warned_contexts", False):
                _notify(on_stage, f"[WARN] {self.preset_name} ignores contexts")
                self._warned_contexts = True

        self.load(on_stage=on_stage)

        import torch
        from whisper.local_backend import _clean_master_text, _get_wav_duration

        results = []
        _notify(on_stage, f"{self.backend_label} 文本生成开始，共 {len(audio_paths)} 条...")

        for idx, audio_path in enumerate(audio_paths):
            normalized_path = str(Path(audio_path).resolve())
            duration = _get_wav_duration(normalized_path)

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
                    predicted_ids = self._model.generate(
                        input_features,
                        do_sample=False,
                        **generate_kwargs,
                    )
                    text = self._processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True,
                    )[0]
                    text = self.post_clean_fn(text)

                master_text = _clean_master_text(text)

                results.append(
                    {
                        "text": master_text,
                        "raw_text": text,
                        "duration": duration,
                        "language": "Japanese",
                        "normalized_path": normalized_path,
                        "log": [f"{self.backend_label} ASR 输出模式: text_only"],
                    }
                )
            except Exception as e:
                _notify(on_stage, f"[ERROR] {self.backend_label} 转录异常 {audio_path}: {e}")
                results.append(
                    {
                        "text": "",
                        "raw_text": "",
                        "duration": duration,
                        "language": "Japanese",
                        "normalized_path": normalized_path,
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
                            {
                                "words": word_dicts,
                                "text": master_text,
                                "raw_text": raw_text,
                                "alignment_mode": "forced_aligner",
                                "duration": duration,
                                "language": language,
                            }
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
                        {
                            "words": [],
                            "text": "",
                            "raw_text": raw_text,
                            "alignment_mode": "empty",
                            "duration": duration,
                            "language": language,
                        },
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
                        {
                            "words": normalized,
                            "text": master_text,
                            "raw_text": raw_text,
                            "alignment_mode": alignment_mode,
                            "duration": duration,
                            "language": language,
                        },
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

