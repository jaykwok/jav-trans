"""`asr.pipeline` is the ASR stage, not a re-export hub for its own submodules.

Two things used to live at the top of that file and both were invisible when
wrong. It reloaded four submodules at import time, which silently forked every
function object they had already handed out; and it re-exported 27 names from
those submodules, most with no caller, which made `asr.pipeline` look like the
place those functions lived. This file pins both back down, because neither
failure mode produces an error - the first produces a monkeypatch that does not
take, the second produces an import that keeps working after the real owner has
moved on.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from asr import chunking, pipeline, result_cache, transcribe
from asr.backends import registry

PIPELINE_SOURCE = Path(pipeline.__file__).read_text(encoding="utf-8")


class TestNoImportTimeReload:
    def test_the_module_does_not_reload_anything(self) -> None:
        """`importlib.reload` re-executes a module in place, so every function
        object it defines is replaced while old references keep pointing at the
        originals. It was here to pick up environment changes in a persistent
        worker; every setting in those four modules is read inside a function,
        so it never bought that and only forked identities."""
        assert "importlib.reload" not in PIPELINE_SOURCE

    @pytest.mark.parametrize(
        "name, owner",
        [
            ("plan_chunk_cuts", chunking),
            ("_get_wav_duration", chunking),
            ("_extract_wav_chunks", chunking),
            ("_transcribe_asr_chunks_text_only", transcribe),
            ("_align_TRANSCRIPTION_results", transcribe),
            ("_group_words_to_segments", transcribe),
            ("_postprocess_segments", transcribe),
            ("get_backend_label", registry),
        ],
    )
    def test_pipeline_shares_one_function_object_with_the_owner(
        self, name: str, owner: object
    ) -> None:
        """The property a reload breaks. Without it, patching
        `asr.transcribe.<name>` leaves `asr.pipeline` calling the pre-reload
        copy - the patch appears to apply and changes nothing."""
        assert getattr(pipeline, name) is getattr(owner, name)


class TestNoDeadReExports:
    # alias that used to sit on `asr.pipeline` -> where the thing really lives.
    # Two of them were renames rather than plain aliases, which is the other
    # cost of a forwarding layer: it keeps an old name alive after the owner has
    # picked a better one.
    RETIRED = {
        "_create_asr_backend": (registry, "_create_asr_backend"),
        "_is_timed_out_result": (result_cache, "_is_timed_out_result"),
        "_checkpointable_text_results": (result_cache, "_cacheable_text_results"),
        "_chunk_duration": (chunking, "_chunk_duration"),
        "ASRWorkerSystemError": (transcribe, "ASRWorkerSystemError"),
        "_strip_punctuation": (transcribe, "_strip_punctuation"),
        "_collapse_repeated_noise": (transcribe, "_collapse_repeated_noise"),
        "_is_low_value_text": (transcribe, "_is_low_value_text"),
        "_clean_segment_text": (transcribe, "_clean_segment_text"),
        "_is_empty_segment_text_result": (transcribe, "_is_empty_segment_text_result"),
        "_empty_alignment_placeholder": (transcribe, "_empty_alignment_placeholder"),
        "_empty_segments_quarantine_placeholder": (
            transcribe,
            "_empty_segments_quarantine_placeholder",
        ),
        "_repair_postprocessed_segment_windows": (
            transcribe,
            "_repair_postprocessed_segment_windows",
        ),
        "_current_asr_backend": (registry, "current_asr_backend"),
        "_QWEN_BACKENDS": (registry, "_QWEN_BACKENDS"),
        "_VALID_ASR_BACKENDS": (registry, "_VALID_ASR_BACKENDS"),
    }

    @pytest.mark.parametrize("alias", sorted(RETIRED))
    def test_a_forwarder_with_no_caller_is_gone(self, alias: str) -> None:
        """Each of these was an alias nothing read. Import them from the module
        that defines them; a second name for the same function is a second place
        to look when it moves."""
        assert not hasattr(pipeline, alias)

    @pytest.mark.parametrize("alias", sorted(RETIRED))
    def test_the_owner_still_has_it(self, alias: str) -> None:
        """The point is that the alias was redundant, not that the function is
        unwanted - so this half must keep passing."""
        owner, name = self.RETIRED[alias]
        assert hasattr(owner, name)


class TestSingleOwnerState:
    def test_the_boundary_signature_lives_in_exactly_one_module(self) -> None:
        """It used to be declared in `asr.chunking` and mirrored into it on every
        write, while nothing in that module ever read it. Two copies of one
        mutable global is a bug waiting for the mirror to be forgotten."""
        assert hasattr(pipeline, "_LAST_BOUNDARY_SIGNATURE")
        assert not hasattr(chunking, "_LAST_BOUNDARY_SIGNATURE")

    def test_setting_it_is_visible_to_the_reader(self) -> None:
        previous = dict(pipeline._LAST_BOUNDARY_SIGNATURE)
        try:
            pipeline._set_last_boundary_signature({"chunking": {"source": "probe"}})
            signature = pipeline._get_asr_runtime_signature()
            assert signature["boundary"]["chunking"]["source"] == "probe"
        finally:
            pipeline._set_last_boundary_signature(previous)

    def test_an_explicit_signature_overrides_the_global(self) -> None:
        """Callers that already know how the audio was cut must not have to
        route it through module state to be heard."""
        signature = pipeline._get_asr_runtime_signature({"chunking": {"source": "given"}})
        assert signature["boundary"]["chunking"]["source"] == "given"


class TestImportsAreStatic:
    def test_every_module_level_import_is_at_the_top(self) -> None:
        """A stray `import logging` halfway down the file was how the reload
        block stayed unnoticed for so long. Function-local imports stay allowed;
        they are how the heavy torch/transformers loads are deferred."""
        tree = ast.parse(PIPELINE_SOURCE)
        import_lines = [
            node.lineno
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        first_definition = min(
            (
                node.lineno
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            ),
            default=10**9,
        )
        assert max(import_lines) < first_definition
