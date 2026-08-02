"""How `ASR_ALIGNMENT_HEAD_PATH` resolves to a file on disk.

The head no longer lives in the git repo - it is a 14.7 MB binary that would
add a full copy to history on every retrain, and it is encoder-specific, so it
belongs next to the encoder it was trained on. It ships in the ASR Hugging Face
repo instead, and the default setting is an `hf:` reference rather than a path.

Two properties these tests exist to hold:

  * the default pins a **commit sha**. Under a moving branch a retrained head
    would change every subtitle's timing on every user's next run, with nothing
    in the log saying so.
  * resolution reaches the network only when the file is not already local.
    Nothing here may make a real request.
"""
from __future__ import annotations

import huggingface_hub
import pytest

from asr import alignment
from core.config import DEFAULT_SETTINGS

DEFAULT_REFERENCE = DEFAULT_SETTINGS["ASR_ALIGNMENT_HEAD_PATH"]


@pytest.fixture
def models_root(monkeypatch, tmp_path):
    """Point models/ at a scratch dir, so nothing touches the real one."""
    from utils import model_paths

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_paths, "MODELS_ROOT", root)
    monkeypatch.setattr(alignment, "_bundled_head_path", lambda filename: "")
    return root


@pytest.fixture
def no_network(monkeypatch, models_root):
    """Any real Hub call is a test failure, not a slow test."""

    def _forbidden(*args, **kwargs):
        raise AssertionError(f"unexpected Hub call: {args} {kwargs}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _forbidden)
    return _forbidden


def test_default_is_a_pinned_hf_reference():
    repo, revision, filename = alignment._parse_hf_reference(DEFAULT_REFERENCE)

    assert repo == DEFAULT_SETTINGS["ASR_BACKEND"], "head must track its own encoder"
    assert filename == "ctc_aligner.pt"
    assert len(revision) == 40 and set(revision) <= set("0123456789abcdef"), (
        f"revision must be a commit sha, not a branch: {revision!r}"
    )


def test_default_still_reads_as_configured():
    """`alignment_head_configured()` is the cheap probe callers gate on."""
    import os

    os.environ["ASR_ALIGNMENT_HEAD_PATH"] = DEFAULT_REFERENCE
    try:
        assert alignment.alignment_head_configured() is True
    finally:
        del os.environ["ASR_ALIGNMENT_HEAD_PATH"]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("hf:ns/repo@abc123#head.pt", ("ns/repo", "abc123", "head.pt")),
        ("hf:ns/repo@abc123", ("ns/repo", "abc123", "ctc_aligner.pt")),
        ("hf:ns/repo#head.pt", ("ns/repo", "", "head.pt")),
        ("hf:ns/repo", ("ns/repo", "", "ctc_aligner.pt")),
        ("  hf:ns/repo@abc123#head.pt  ", ("ns/repo", "abc123", "head.pt")),
    ],
)
def test_reference_parsing(reference, expected):
    assert alignment._parse_hf_reference(reference) == expected


@pytest.mark.parametrize("reference", ["hf:", "hf:@abc123", "hf:ns/repo#"])
def test_malformed_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        alignment._parse_hf_reference(reference)


def test_plain_path_is_returned_untouched(no_network, tmp_path):
    """A local path keeps working as an expert override, with no Hub call."""
    local = tmp_path / "my_head.pt"

    assert alignment.resolve_alignment_head_path(str(local)) == str(local)
    assert alignment.resolve_alignment_head_path("") == ""


def _write_head(models_root, revision: str | None) -> "object":
    head = models_root / "ctc_aligner.pt"
    head.write_bytes(b"weights")
    if revision is not None:
        alignment._revision_marker(head).write_text(revision, encoding="utf-8")
    return head


def test_downloaded_head_resolves_from_models_without_downloading(
    no_network, models_root
):
    _, revision, _ = alignment._parse_hf_reference(DEFAULT_REFERENCE)
    head = _write_head(models_root, revision)

    assert alignment.resolve_alignment_head_path(DEFAULT_REFERENCE) == str(head)


def test_head_from_a_different_revision_is_refetched(monkeypatch, models_root):
    """Re-pinning the sha must not load the old head under the same name."""
    _write_head(models_root, "0" * 40)
    calls: list[dict] = []

    def _download(**kwargs):
        calls.append(kwargs)
        return str(models_root / "ctc_aligner.pt")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    alignment.resolve_alignment_head_path(DEFAULT_REFERENCE)

    assert len(calls) == 1, "a stale revision must trigger a fresh download"


def test_head_without_a_revision_marker_is_refetched(monkeypatch, models_root):
    """An interrupted download leaves the file but no marker; do not trust it."""
    _write_head(models_root, revision=None)
    calls: list[dict] = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kw: calls.append(kw) or str(models_root / "ctc_aligner.pt"),
    )

    alignment.resolve_alignment_head_path(DEFAULT_REFERENCE)

    assert len(calls) == 1


def test_download_lands_in_models_at_the_pinned_revision(monkeypatch, models_root):
    calls: list[dict] = []

    def _download(**kwargs):
        calls.append(kwargs)
        target = models_root / kwargs["filename"]
        target.write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    resolved = alignment.resolve_alignment_head_path(DEFAULT_REFERENCE)
    repo, revision, filename = alignment._parse_hf_reference(DEFAULT_REFERENCE)

    assert resolved == str(models_root / filename)
    assert calls == [
        {
            "repo_id": repo,
            "filename": filename,
            "revision": revision,
            "local_dir": str(models_root),
        }
    ], "must download into models/, not the Hub cache under tmp/"
    marker = alignment._revision_marker(models_root / filename)
    assert marker.read_text(encoding="utf-8") == revision

    # The marker is what makes the second call free.
    calls.clear()
    assert alignment.resolve_alignment_head_path(DEFAULT_REFERENCE) == resolved
    assert calls == []


def test_download_disabled_yields_nothing_rather_than_fetching(no_network, models_root):
    """The cache-signature path must never start a download of its own."""
    assert alignment.resolve_alignment_head_path(DEFAULT_REFERENCE, download=False) == ""


def test_bundled_copy_wins_over_the_hub(monkeypatch, no_network, tmp_path):
    """The packaged build has no network guarantee on a user's first run."""
    bundled = tmp_path / "bundled" / "ctc_aligner.pt"
    bundled.parent.mkdir()
    bundled.write_bytes(b"weights")
    monkeypatch.setattr(alignment, "_bundled_head_path", lambda filename: str(bundled))

    assert alignment.resolve_alignment_head_path(DEFAULT_REFERENCE) == str(bundled)


def test_finalize_signature_keys_off_the_resolved_bytes(monkeypatch, tmp_path):
    """The finalize cache must follow the head's content, not its reference.

    Swapping heads has to invalidate cached word timings; otherwise a rerun
    reports the old head's timeline as the new head's output.
    """
    import hashlib

    from asr import result_cache

    head = tmp_path / "ctc_aligner.pt"
    head.write_bytes(b"weights")
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", DEFAULT_REFERENCE)
    monkeypatch.setattr(
        alignment, "resolve_alignment_head_path", lambda ref, download=True: str(head)
    )

    signature = result_cache.finalize_signature()

    assert signature is not None
    assert signature["alignment_head"] == {
        "sha256": hashlib.sha256(b"weights").hexdigest()
    }


def test_unresolvable_reference_disables_the_finalize_cache(monkeypatch):
    """No head on disk yet means no signature - not a crash, and not a stale key."""
    from asr import result_cache

    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", DEFAULT_REFERENCE)
    monkeypatch.setattr(
        alignment, "resolve_alignment_head_path", lambda ref, download=True: ""
    )

    assert result_cache.finalize_signature() is None
