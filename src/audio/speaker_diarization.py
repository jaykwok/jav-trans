import os
import gc
import logging
import re
import numpy as np
import torch
import torchaudio
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict

from utils.model_paths import MODELS_ROOT, model_dir_name

logger = logging.getLogger(__name__)
_SPEAKER_BGM_RE = re.compile(r"^[ぁ-ゟ\s、。！？…ー～「」『』・\(\)（）]+$")


def _env_val(key: str, default):
    val = os.getenv(key, str(default))
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


def _import_classifier(source: str, savedir: str, run_opts: dict):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source=source, savedir=savedir, run_opts=run_opts
    )


def diarize_segments(
    segments: list[dict],
    audio_path,
    *,
    model_dir: str = "",
    min_duration_s: float | None = None,
    cluster_threshold: float | None = None,
    max_clusters: int | None = None,
) -> list[dict]:
    """Return segments with 'speaker' field (S0/S1/... or None for BGM/short).

    Falls back to original segments unchanged on any error.
    Controlled by env EXPERIMENTAL_SPEAKER_DIARIZATION (default 0 = off).
    """
    if not _env_val("EXPERIMENTAL_SPEAKER_DIARIZATION", 0):
        return segments

    _min_dur = min_duration_s if min_duration_s is not None else _env_val("SPEAKER_MIN_DURATION", 0.5)
    _threshold = cluster_threshold if cluster_threshold is not None else _env_val("SPEAKER_CLUSTER_THRESHOLD", 0.5)
    _max_clusters = max_clusters if max_clusters is not None else _env_val("SPEAKER_MAX_CLUSTERS", 5)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = _env_val("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        savedir = model_dir or str(MODELS_ROOT / model_dir_name(source))
        logger.info("Speaker diarization: loading ECAPA model to %s", device)
        classifier = _import_classifier(
            source=source,
            savedir=savedir,
            run_opts={"device": device},
        )
        classifier.eval()

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.to(device)
        total_samples = waveform.shape[1]

        embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []
        skipped: dict[int, str] = {}

        for i, seg in enumerate(segments):
            dur = seg.get("end", 0.0) - seg.get("start", 0.0)
            ja_text = (seg.get("text") or seg.get("ja") or "").strip()

            if dur < _min_dur:
                skipped[i] = "short"
                continue
            if ja_text and _SPEAKER_BGM_RE.fullmatch(ja_text):
                skipped[i] = "bgm"
                continue

            start_s = int(seg.get("start", 0.0) * sr)
            end_s = int(seg.get("end", 0.0) * sr)
            end_s = min(end_s, total_samples)
            if end_s <= start_s:
                skipped[i] = "short"
                continue

            clip = waveform[:, start_s:end_s]
            # torch.no_grad() is mandatory — without it PyTorch keeps full computation
            # graph for all 300+ embeddings and will OOM on 8GB VRAM
            with torch.no_grad():
                emb = classifier.encode_batch(clip)  # shape (1, 1, dim)
            emb_np = emb.squeeze().cpu().float().numpy()
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            embeddings.append(emb_np)
            valid_indices.append(i)

        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = [dict(s) for s in segments]

        if len(embeddings) < 2:
            for i in valid_indices:
                result[i]["speaker"] = "S0"
            for i in skipped:
                result[i]["speaker"] = None
            return result

        X = np.stack(embeddings)
        clust = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=_threshold,
        )
        labels = clust.fit_predict(X)

        # Cap clusters: remap smallest ones to nearest kept cluster
        n_clusters = len(set(labels))
        if n_clusters > _max_clusters:
            counts = Counter(labels)
            keep = {lbl for lbl, _ in counts.most_common(_max_clusters)}
            kept_idx = [i for i, lbl in enumerate(labels) if lbl in keep]
            kept_X = X[kept_idx]
            kept_lbls = labels[kept_idx]
            for i, lbl in enumerate(labels):
                if lbl not in keep:
                    dists = 1.0 - (kept_X @ X[i])
                    labels[i] = kept_lbls[int(np.argmin(dists))]

        # Rename by cluster size: largest = S0
        counts = Counter(labels)
        rank_map = {lbl: f"S{rank}" for rank, (lbl, _) in enumerate(counts.most_common())}

        for idx, seg_i in enumerate(valid_indices):
            result[seg_i]["speaker"] = rank_map[labels[idx]]
        for seg_i in skipped:
            result[seg_i]["speaker"] = None

        # Backfill short segments with nearest valid neighbor's speaker
        for i in range(len(result)):
            if result[i].get("speaker") is None and skipped.get(i) == "short" and valid_indices:
                mid = (result[i].get("start", 0.0) + result[i].get("end", 0.0)) / 2.0
                nearest = min(
                    valid_indices,
                    key=lambda j: abs((result[j].get("start", 0.0) + result[j].get("end", 0.0)) / 2.0 - mid),
                )
                result[i]["speaker"] = result[nearest]["speaker"]

        return result

    except Exception as exc:
        logger.warning("Speaker diarization failed, skipping: %s", exc)
        return segments


def build_speakers_report(segments: list[dict]) -> dict:
    """Build speakers.json summary from diarized segments."""
    speaker_stats: dict[str, dict] = defaultdict(lambda: {"n_segments": 0, "total_duration_s": 0.0})
    bgm_count = 0
    other_short = 0

    for s in segments:
        spk = s.get("speaker")
        dur = s.get("end", 0.0) - s.get("start", 0.0)
        if spk is None:
            ja = (s.get("text") or s.get("ja") or "").strip()
            if ja and _SPEAKER_BGM_RE.fullmatch(ja):
                bgm_count += 1
            else:
                other_short += 1
        else:
            speaker_stats[spk]["n_segments"] += 1
            speaker_stats[spk]["total_duration_s"] += dur

    for st in speaker_stats.values():
        n = st["n_segments"]
        st["avg_segment_s"] = round(st["total_duration_s"] / n, 2) if n > 0 else 0.0
        st["total_duration_s"] = round(st["total_duration_s"], 2)

    return {
        "n_speakers": len(speaker_stats),
        "note": (
            "ECAPA trained on VoxCeleb; JAV extreme vocalizations (moans/cries) "
            "may cause same-speaker fragmentation into multiple S* labels"
        ),
        "speakers": dict(speaker_stats),
        "unassigned": {"bgm": bgm_count, "other_short": other_short},
    }

