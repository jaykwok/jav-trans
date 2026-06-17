"""CueQC Mamba v3-Fusion runtime adapter.

Loads a ``cueqc_mamba_checkpoint_v3_fusion`` checkpoint, rebuilds
``CueQCMambaV3Fusion``, and exposes ``decide()`` that emits the runtime contract
dict consumed by ``pipeline._run_cueqc_shadow``.

The refiner never loads its own Qwen3-ASR — at runtime the caller passes an
``AsrInternalsCapturer`` that wraps the already-loaded ``LocalAsrBackend.model``
(see v3-Fusion §5.2: no second ASR model in VRAM). All failures fall back to
``keep`` (v3-Fusion §2.4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import os

CUEQC_MAMBA_CHECKPOINT_SCHEMA = "cueqc_mamba_checkpoint_v3_fusion"
MODE_TAG = "cueqc_mamba_v3_fusion"
FALLBACK_MODE_TAG = "fallback_keep"


def _file_sha1(path: Path) -> str:
    import hashlib

    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _short_detail(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _fallback_keep(
    reason: str,
    cluster_id: str = "unclustered",
    *,
    stage: str = "",
    detail: Any = "",
) -> dict[str, Any]:
    detail_text = _short_detail(detail)
    reasons = [reason]
    if detail_text:
        reasons.append(f"{reason}:{detail_text}")
    return {
        "schema": "cueqc_shadow_v1",
        "model_version": MODE_TAG,
        "decision_version": "cueqc_display_binary_v1",
        "mode": FALLBACK_MODE_TAG,
        "display_hint": "keep",
        "cluster_id": cluster_id,
        "confidence": 1.0,
        "display_prob_keep": 1.0,
        "display_prob_drop": 0.0,
        "fallback_stage": stage or "unknown",
        "fallback_detail": detail_text,
        "reasons": reasons,
    }


class CueQCRefinerV3Fusion:
    """Runtime wrapper around a trained CueQC Mamba v3-Fusion checkpoint."""

    def __init__(self, *, checkpoint: dict[str, Any], path: Path, device: str = "auto") -> None:
        import torch

        schema = str(checkpoint.get("schema") or "")
        if schema != CUEQC_MAMBA_CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported CueQC checkpoint schema: {schema!r}")
        from asr.cueqc_model import CueQCMambaV3Fusion

        model_config = dict(checkpoint.get("model_config") or {})
        # Only keep CueQCMambaV3Fusion constructor kwargs.
        valid = {
            "asr_dim", "token_dim", "decoder_dim", "structured_dim",
            "hidden_size", "num_layers", "state_size", "num_heads", "head_dim",
            "n_groups", "chunk_size", "mlp_dim", "dropout",
        }
        model_config = {k: v for k, v in model_config.items() if k in valid}
        self.model = CueQCMambaV3Fusion(**model_config)
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError("checkpoint missing 'state_dict'")
        self.model.load_state_dict(state_dict)

        normalized = (device or "auto").strip().lower()
        if normalized == "auto":
            normalized = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(normalized)
        self.model.to(self.device)
        self.model.eval()

        self.path = path
        self.sha1 = _file_sha1(path)
        self.decision_version = str(
            checkpoint.get("decision_version")
            or "cueqc_display_binary_v1"
        )
        self.feature_config = dict(checkpoint.get("feature_config") or {})

        norm = checkpoint.get("normalization") or {}
        self.asr_mean = np.asarray(norm.get("asr_mean", []), dtype=np.float32)
        self.asr_std = np.asarray(norm.get("asr_std", []), dtype=np.float32)
        self.token_mean = np.asarray(norm.get("token_mean", []), dtype=np.float32)
        self.token_std = np.asarray(norm.get("token_std", []), dtype=np.float32)
        self.decoder_mean = np.asarray(norm.get("decoder_mean", []), dtype=np.float32)
        self.decoder_std = np.asarray(norm.get("decoder_std", []), dtype=np.float32)
        self.structured_mean = np.asarray(norm.get("structured_mean", []), dtype=np.float32)
        self.structured_std = np.asarray(norm.get("structured_std", []), dtype=np.float32)

        decision = checkpoint.get("decision_config") or {}
        self.decision_config = dict(decision)
        self.drop_threshold = float(decision.get("drop_threshold", 0.85))
        self.drop_threshold_profile = decision.get("drop_threshold_profile") or {}
        self.fallback_policy = str(decision.get("fallback_policy", "keep"))

    def signature(self) -> dict[str, Any]:
        return {
            "type": "cueqc_mamba",
            "schema": CUEQC_MAMBA_CHECKPOINT_SCHEMA,
            "path": str(self.path),
            "sha1": self.sha1,
            "decision_version": self.decision_version,
            "drop_threshold": self.drop_threshold,
            "drop_threshold_profile": self.drop_threshold_profile,
            "feature_config": self.feature_config,
        }

    # --------------------------------------------------------- feature build
    def _build_sample(self, candidate: dict[str, Any], internals: dict[str, Any]) -> dict[str, np.ndarray]:
        from asr.cueqc_features import (
            build_decoder_stats,
            build_structured_features,
            build_token_trace,
        )

        asr_frames = internals["asr_frames"]
        if asr_frames.shape[0] == 0:
            raise ValueError("empty asr_frames")

        token_trace = build_token_trace(
            token_logprobs=internals["token_logprobs"],
            token_entropies=internals["token_entropies"],
            token_margins=internals["token_top1_top2_margins"],
            decoded_tokens=internals["decoded_tokens"],
            has_timestamps=bool(internals.get("has_timestamps", False)),
        )
        start_s = float(candidate.get("start", 0.0))
        end_s = float(candidate.get("end", start_s))
        duration_s = max(float(candidate.get("duration_s", end_s - start_s)), 1e-6)
        decoder_stats = build_decoder_stats(
            token_trace=token_trace,
            text=str(candidate.get("text") or ""),
            duration_s=duration_s,
            has_timestamps=bool(internals.get("has_timestamps", False)),
        )
        asr_confidence = float(np.mean(internals["token_logprobs"])) if internals["token_logprobs"].size else None
        structured = build_structured_features(
            candidate,
            n_tokens=int(internals["token_ids"].shape[0]),
            asr_confidence=asr_confidence,
        )
        return {
            "asr_frames": np.ascontiguousarray(asr_frames, dtype=np.float32),
            "token_trace": np.ascontiguousarray(token_trace, dtype=np.float32),
            "decoder_stats": np.ascontiguousarray(decoder_stats, dtype=np.float32),
            "structured": np.ascontiguousarray(structured, dtype=np.float32),
        }

    def _normalize(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def z(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            if mean.size != arr.shape[-1] or std.size != arr.shape[-1]:
                return arr  # mismatched layout -> leave raw (avoid crash)
            out = (arr - mean) / np.maximum(std, 1e-6)
            return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

        return {
            "asr_frames": z(sample["asr_frames"], self.asr_mean, self.asr_std),
            "token_trace": z(sample["token_trace"], self.token_mean, self.token_std),
            "decoder_stats": z(sample["decoder_stats"].reshape(1, -1), self.decoder_mean, self.decoder_std).reshape(-1),
            "structured": z(sample["structured"].reshape(1, -1), self.structured_mean, self.structured_std).reshape(-1),
        }

    @staticmethod
    def _pad_batch(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        max_t = max(int(s.shape[0]) for s in seqs)
        d = int(seqs[0].shape[1])
        out = np.zeros((len(seqs), max_t, d), dtype=np.float32)
        mask = np.zeros((len(seqs), max_t), dtype=np.float32)
        for i, s in enumerate(seqs):
            t = int(s.shape[0])
            out[i, :t] = s
            mask[i, :t] = 1.0
        return out, mask

    # ----------------------------------------------------------------- decide
    def decide(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        asr_internals: Sequence[dict[str, Any]] | None = None,
        capturer=None,
        audio_path_by_idx: Sequence[str | None] | None = None,
        default_audio_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run batched inference for the given candidates.

        Two capture modes (worker-agnostic):
        * ``asr_internals`` — pre-captured internals list (one per candidate),
          produced by ``backend.capture_asr_internals(...)`` (handles both inline
          and subprocess worker modes). Preferred at runtime.
        * ``capturer`` — an ``AsrInternalsCapturer`` for offline/legacy capture.
          Used only when ``asr_internals`` is not provided.

        Each candidate gets a decision dict; any per-candidate failure falls
        back to ``keep``.
        """
        import torch

        n = len(candidates)
        if n == 0:
            return []

        use_pre = asr_internals is not None and len(asr_internals) == n

        # Build features per candidate, recording which ones succeeded.
        norm_samples: list[dict[str, np.ndarray]] = []
        sample_indices: list[int] = []
        failures: dict[int, dict[str, str]] = {}
        for i, cand in enumerate(candidates):
            try:
                if use_pre:
                    internals = asr_internals[i]  # type: ignore[index]
                    if not isinstance(internals, dict) or not internals.get("ok"):
                        detail = ""
                        if isinstance(internals, dict):
                            detail = str(internals.get("error") or internals.get("detail") or "")
                        failures[i] = {"stage": "capture", "reason": "cueqc_capture_error", "detail": detail}
                        continue
                else:
                    if capturer is None:
                        failures[i] = {
                            "stage": "capture_request",
                            "reason": "cueqc_capture_unavailable",
                            "detail": "capturer is not configured",
                        }
                        continue
                    wav = (audio_path_by_idx[i] if audio_path_by_idx else default_audio_path)
                    if not wav:
                        failures[i] = {
                            "stage": "capture_request",
                            "reason": "cueqc_capture_audio_missing",
                            "detail": "empty audio path",
                        }
                        continue
                    start_s = float(cand.get("start", 0.0))
                    end_s = float(cand.get("end", start_s))
                    internals = capturer.extract(wav, str(cand.get("text") or ""), start_s=start_s, end_s=end_s)
                sample = self._build_sample(cand, internals)
                norm_samples.append(self._normalize(sample))
                sample_indices.append(i)
            except Exception as exc:  # noqa: BLE001 - runtime must not crash the pipeline
                failures[i] = {
                    "stage": "feature",
                    "reason": "cueqc_feature_error",
                    "detail": repr(exc),
                }

        # Build decision list; failed candidates get fallback keep.
        decisions: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]
        for i, failure in failures.items():
            cand = candidates[i]
            decisions[i] = _fallback_keep(
                failure["reason"],
                str(cand.get("cluster_id", "unclustered")),
                stage=failure["stage"],
                detail=failure.get("detail", ""),
            )

        try:
            batch_size = max(1, int(float(os.getenv("CUEQC_INFERENCE_BATCH_SIZE", "64"))))
        except (TypeError, ValueError):
            batch_size = 64

        for start in range(0, len(norm_samples), batch_size):
            batch_samples = norm_samples[start : start + batch_size]
            batch_candidate_indices = sample_indices[start : start + batch_size]
            asr_batch, asr_mask = self._pad_batch([s["asr_frames"] for s in batch_samples])
            tok_batch, tok_mask = self._pad_batch([s["token_trace"] for s in batch_samples])
            dec_batch = np.stack([s["decoder_stats"] for s in batch_samples])
            struct_batch = np.stack([s["structured"] for s in batch_samples])
            try:
                with torch.inference_mode():
                    logits = self.model(
                        asr_frames=torch.from_numpy(asr_batch).to(self.device),
                        asr_mask=torch.from_numpy(asr_mask).to(self.device),
                        token_trace=torch.from_numpy(tok_batch).to(self.device),
                        token_mask=torch.from_numpy(tok_mask).to(self.device),
                        decoder_stats=torch.from_numpy(dec_batch).to(self.device),
                        structured=torch.from_numpy(struct_batch).to(self.device),
                    )
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            except Exception as exc:  # noqa: BLE001 - inference failure -> fallback keep for this batch
                if str(self.device).startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                for i in batch_candidate_indices:
                    cand = candidates[i]
                    decisions[i] = _fallback_keep(
                        "cueqc_inference_error",
                        str(cand.get("cluster_id", "unclustered")),
                        stage="inference",
                        detail=repr(exc),
                    )
                continue

            from asr.cueqc_thresholds import resolve_drop_threshold

            for local_j, i in enumerate(batch_candidate_indices):
                cand = candidates[i]
                cluster_id = str(cand.get("cluster_id", "unclustered"))
                p_drop = float(probs[local_j, 0])
                p_keep = float(probs[local_j, 1])
                threshold, threshold_info = resolve_drop_threshold(
                    self.decision_config,
                    text=cand.get("text", ""),
                    default=self.drop_threshold,
                )
                is_drop = p_drop >= threshold
                display = "drop" if is_drop else "keep"
                confidence = round(p_drop if is_drop else p_keep, 4)
                decisions[i] = {
                    "schema": "cueqc_shadow_v1",
                    "model_version": MODE_TAG,
                    "decision_version": self.decision_version,
                    "mode": MODE_TAG,
                    "display_hint": display,
                    "cluster_id": cluster_id,
                    "confidence": confidence,
                    "display_prob_keep": round(p_keep, 4),
                    "display_prob_drop": round(p_drop, 4),
                    "drop_threshold": round(threshold, 4),
                    "threshold_profile": threshold_info,
                    "reasons": [f"cueqc_mamba_v3:{display}:p_drop={p_drop:.3f}:threshold={threshold:.3f}"],
                }
        for i, decision in enumerate(decisions):
            if decision is None:
                cand = candidates[i]
                decisions[i] = _fallback_keep(
                    "cueqc_unknown_runtime_error",
                    str(cand.get("cluster_id", "unclustered")),
                    stage="unknown",
                )
        return decisions


def load_cueqc_mamba_checkpoint(path: str | Path, *, device: str = "auto") -> CueQCRefinerV3Fusion:
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CueQC checkpoint not found: {p}")
    checkpoint = torch.load(p, map_location="cpu", weights_only=False)
    return CueQCRefinerV3Fusion(checkpoint=checkpoint, path=p, device=device)
