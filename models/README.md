# Local Models

Runtime models are downloaded here on first use and are intentionally not
tracked by git.

Each download goes to the canonical `models/<namespace>-<repo>/` directory: the
HuggingFace repo id with its slash replaced by a hyphen.

`hub/` and `xet/` are HuggingFace download caches and belong here because they
hold real weights; deleting them costs a re-download. The torch hub cache
(`tmp/cache/torch`) lives under `tmp/` instead, because it can be regenerated.

The default CTC alignment head is `ctc_aligner_jav_vocalisation_v3.pt`, which
adds a three-class frame head (silence / vocalisation / speech) beside the CTC
classifier. The earlier `ctc_aligner_jav_vocalisation_v2.pt` and the general
`ctc_aligner.pt` remain separate Hugging Face artifacts; they are downloaded
only when `ASR_ALIGNMENT_HEAD_PATH` selects one explicitly.

Each head file has a `.revision` sidecar recording the commit it was fetched
at. When the default sha in `src/core/config.py` is re-pinned, the new head is
fetched instead of the old one being loaded silently under the same name.
