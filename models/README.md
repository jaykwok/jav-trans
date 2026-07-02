# Local Models

Runtime models are downloaded here on first use and are intentionally not tracked by git.

Downloads must use the canonical `models/<namespace>-<repo>/` directory, replacing the slash in the HuggingFace repo id with a hyphen.

HuggingFace hub/xet cache (`tmp/cache/hf/hub`) and torch hub cache (`tmp/cache/torch`) are runtime artifacts outside this directory. If `hub/`, `xet/`, `torch/`, or `models--*` appear here, treat them as cleanup candidates rather than supported model folders.
