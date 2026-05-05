# Local Models

Runtime models are downloaded here on first use and are intentionally not tracked by git.

Downloads must use the canonical `models/<namespace>-<repo>/` directory, replacing the slash in the HuggingFace repo id with a hyphen.

HuggingFace hub/xet cache and torch hub cache are runtime artifacts under `temp/`, not model directories. If `hub/`, `xet/`, `torch/`, `models--*`, or `audio-separator/` appear here, treat them as cleanup candidates rather than supported model folders.
