# Local Models

Runtime models are downloaded here on first use and are intentionally not tracked by git.

Downloads must use the canonical `models/<namespace>-<repo>/` directory, replacing the slash in the HuggingFace repo id with a hyphen.

`hub/` and `xet/` are the HuggingFace download caches and belong here: they hold real weights, so deleting them costs a re-download. The torch hub cache (`tmp/cache/torch`) stays under `tmp/` because it is genuinely regenerable.

The project default CTC file is `ctc_aligner_jav_vocalisation_v3.pt`, which
carries a three-class frame head (silence / vocalisation / speech) beside the
CTC classifier. The previous `ctc_aligner_jav_vocalisation_v2.pt` and the
original general `ctc_aligner.pt` remain separate Hugging Face artifacts and are
downloaded only when `ASR_ALIGNMENT_HEAD_PATH` explicitly selects one; neither
was overwritten.

Each file has a `.revision` sidecar recording which commit it was fetched at.
Re-pinning the default sha in `config.py` therefore fetches the new head instead
of silently loading the old one under a name that no longer matches.
