"""The three-class frame objective, and the labels that feed it.

The class that did not exist before v2 is `vocalisation`. Until it did, the head
had one way to say "not a word" and moaning shared it with silence, so every
dose of "moaning is blank" also taught it that breathy speech is blank - three
falsified ablations and 21 cues whose blank share was exactly 1.0000. These
tests pin the two properties that make the replacement worth anything: the loss
must read the *frame head* rather than the CTC distribution, and it must not
weight `vocalisation` by how rare it is.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    FRAME_CLASS_SILENCE,
    FRAME_CLASS_SPEECH,
    FRAME_CLASS_VOCALISATION,
    FRAME_CLASSES,
)
from tools.align.frame_teacher_supervision import (  # noqa: E402
    IGNORE_LABEL,
    balanced_frame_class_loss,
)
from tools.align.train_ctc_aligner import load_frame_class_labels  # noqa: E402

torch = pytest.importorskip("torch")


def _log_probs(rows: list[list[float]]):
    return torch.log(torch.tensor([rows], dtype=torch.float32))


class TestTheThreeClassLoss:
    def test_ignored_frames_contribute_nothing(self) -> None:
        confident = _log_probs([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        labels = torch.tensor([[FRAME_CLASS_SILENCE, IGNORE_LABEL]])

        loss, counts = balanced_frame_class_loss(confident, labels, torch)

        assert counts == {
            "class_0_frames": 1,
            "class_1_frames": 0,
            "class_2_frames": 0,
        }
        assert float(loss) == pytest.approx(-np.log(0.8), abs=1e-5)

    def test_a_rare_class_is_not_weighted_by_how_rare_it_is(self) -> None:
        """The whole point of the per-class mean.

        `vocalisation` is roughly a tenth of the labelled frames. Under a frame
        mean it would contribute a tenth of the gradient - the class the change
        exists for, drowned by the two that were already working.
        """
        # Nine confident silence frames and one badly wrong vocalisation frame.
        rows = [[0.98, 0.01, 0.01]] * 9 + [[0.98, 0.01, 0.01]]
        labels = torch.tensor([[FRAME_CLASS_SILENCE] * 9 + [FRAME_CLASS_VOCALISATION]])

        loss, counts = balanced_frame_class_loss(_log_probs(rows), labels, torch)

        assert counts["class_1_frames"] == 1
        # The single wrong frame is half the loss, not a tenth of it.
        expected = (-np.log(0.98) + -np.log(0.01)) / 2
        assert float(loss) == pytest.approx(expected, abs=1e-4)

    def test_it_reads_the_frame_head_not_the_ctc_distribution(self) -> None:
        """Its two-class predecessor asked the CTC output whether a frame was
        blank, which was the only question available when moaning had no class -
        and is why more of that supervision made the head deafer rather than
        better. A tensor the width of the CTC vocabulary is refused."""
        ctc_shaped = torch.zeros((1, 4, 2603))
        labels = torch.zeros((1, 4), dtype=torch.long)

        with pytest.raises(ValueError, match="expected 3"):
            balanced_frame_class_loss(ctc_shaped, labels, torch)

    def test_a_batch_with_no_labelled_frame_yields_a_zero_that_still_backprops(
        self,
    ) -> None:
        frames = torch.zeros((1, 3, 3), requires_grad=True)
        labels = torch.full((1, 3), IGNORE_LABEL)

        loss, counts = balanced_frame_class_loss(frames, labels, torch)
        loss.backward()

        assert float(loss.detach()) == 0.0
        assert sum(counts.values()) == 0
        assert frames.grad is not None

    def test_mismatched_label_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame label shape"):
            balanced_frame_class_loss(
                torch.zeros((1, 4, 3)), torch.zeros((1, 5), dtype=torch.long), torch
            )


class TestLoadingTheLabelArchive:
    @staticmethod
    def _archive(tmp_path: Path, name: str, entries: dict[str, list[int]]) -> str:
        path = tmp_path / name
        np.savez_compressed(
            path, **{key: np.array(value, dtype=np.int8) for key, value in entries.items()}
        )
        return str(path.with_suffix(".npz")) if path.suffix != ".npz" else str(path)

    def test_keys_carry_the_cache_so_ids_cannot_collide(self, tmp_path: Path) -> None:
        """The corpora number their clips independently; matching by id alone
        would hand one cache's labels to another cache's audio."""
        archive = self._archive(
            tmp_path,
            "labels.npz",
            {
                "anime-nsfw/shared": [FRAME_CLASS_VOCALISATION, FRAME_CLASS_SPEECH],
                "galgame-teacher/shared": [FRAME_CLASS_SPEECH, FRAME_CLASS_SPEECH],
            },
        )
        rows = [
            {"audio_id": "shared", "cache_index": 0},
            {"audio_id": "shared", "cache_index": 1},
        ]

        labels, summary = load_frame_class_labels(
            rows, ["anime-nsfw", "galgame-teacher"], archives=[archive]
        )

        assert labels[(0, "shared")].tolist() == [
            FRAME_CLASS_VOCALISATION,
            FRAME_CLASS_SPEECH,
        ]
        assert labels[(1, "shared")].tolist() == [FRAME_CLASS_SPEECH] * 2
        assert summary["train_rows_covered"] == 2

    def test_partial_coverage_is_reported_not_refused(self, tmp_path: Path) -> None:
        """Unlike the teacher archives, no source covers every corpus: L3 has no
        vocalisation labels by design, and a clip that failed the L1 quality gate
        has none on purpose. Requiring coverage would make the gate
        all-or-nothing."""
        archive = self._archive(
            tmp_path, "partial.npz", {"anime-nsfw/a": [FRAME_CLASS_SPEECH]}
        )
        rows = [
            {"audio_id": "a", "cache_index": 0},
            {"audio_id": "b", "cache_index": 0},
        ]

        labels, summary = load_frame_class_labels(
            rows, ["anime-nsfw"], archives=[archive]
        )

        assert set(labels) == {(0, "a")}
        assert summary["train_rows_without_frame_classes"] == 1

    def test_labels_for_a_cache_the_run_did_not_load_are_counted_and_dropped(
        self, tmp_path: Path
    ) -> None:
        archive = self._archive(
            tmp_path,
            "extra.npz",
            {"anime-nsfw/a": [FRAME_CLASS_SPEECH], "anime-sfw/b": [FRAME_CLASS_SILENCE]},
        )
        rows = [{"audio_id": "a", "cache_index": 0}]

        _labels, summary = load_frame_class_labels(
            rows, ["anime-nsfw"], archives=[archive]
        )

        assert summary["rows_in_caches_not_loaded"] == {"anime-sfw": 1}

    def test_an_archive_with_nothing_labelled_stops_the_run(
        self, tmp_path: Path
    ) -> None:
        archive = self._archive(
            tmp_path, "empty.npz", {"anime-nsfw/a": [IGNORE_LABEL, IGNORE_LABEL]}
        )

        with pytest.raises(SystemExit, match="no labelled frames"):
            load_frame_class_labels(
                [{"audio_id": "a", "cache_index": 0}], ["anime-nsfw"], archives=[archive]
            )

    def test_the_class_shares_are_reported_for_the_mixture_actually_loaded(
        self, tmp_path: Path
    ) -> None:
        """`vocalisation` is the minority class and the run has to say by how
        much, because that is what decides whether the per-class mean is doing
        the work it was chosen for."""
        archive = self._archive(
            tmp_path,
            "mix.npz",
            {
                "anime-nsfw/a": [FRAME_CLASS_SILENCE] * 5
                + [FRAME_CLASS_VOCALISATION]
                + [FRAME_CLASS_SPEECH] * 4
            },
        )

        _labels, summary = load_frame_class_labels(
            [{"audio_id": "a", "cache_index": 0}], ["anime-nsfw"], archives=[archive]
        )

        assert summary["class_share_of_labelled"] == {
            "silence": 0.5,
            "vocalisation": 0.1,
            "speech": 0.4,
        }
        assert list(summary["frames"]) == list(FRAME_CLASSES)


class TestTheTwoObjectivesCannotBothRun:
    def test_the_trainer_refuses_the_pair(self) -> None:
        """They supervise the same frames with different class systems, and the
        two-class one reads the CTC output while the three-class one reads its
        own head. Together they train a head whose two answers about a moan
        disagree by construction."""
        source = (
            PROJECT_ROOT / "tools" / "align" / "train_ctc_aligner.py"
        ).read_text(encoding="utf-8")

        assert "--frame-class-labels replaces the two-class frame teacher" in source

    def test_a_frame_class_run_writes_a_v2_checkpoint_naming_its_classes(self) -> None:
        source = (
            PROJECT_ROOT / "tools" / "align" / "train_ctc_aligner.py"
        ).read_text(encoding="utf-8")

        assert 'payload["frame_classes"] = list(FRAME_CLASSES)' in source
        assert "ALIGNMENT_MODEL_SCHEMA_V2" in source
