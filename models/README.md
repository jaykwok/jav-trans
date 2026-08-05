# Local Models

Runtime models are downloaded here on first use and are intentionally not tracked by git.

Downloads must use the canonical `models/<namespace>-<repo>/` directory, replacing the slash in the HuggingFace repo id with a hyphen.

`hub/` and `xet/` are the HuggingFace download caches and belong here: they hold real weights, so deleting them costs a re-download. The torch hub cache (`tmp/cache/torch`) stays under `tmp/` because it is genuinely regenerable.
