"""Serializing a path is a path operation; publishing one is an atomic move.

Both were done by string: the project root was substituted anywhere it appeared
in any string of the payload, and the staging file was named after the target
plus the pid. Each has a failure that only shows up next to something the
project does not own - a sibling directory, or a second writer.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from pipeline import artifacts as artifacts_module
from pipeline import artifact_store
from utils.model_paths import PROJECT_ROOT


def _sibling_of(root: Path) -> Path:
    """A directory next to the root whose name *starts* with the root's name."""
    return root.parent / f"{root.name}-backup"


class TestOnlyPathsUnderTheRootAreMadeRelative:
    def test_a_sibling_of_the_root_keeps_its_name(self, tmp_path) -> None:
        external = _sibling_of(tmp_path) / "sample-a.srt"

        serialized = artifact_store.project_relative(external, project_root=tmp_path)

        # Not `-backup/sample-a.srt`: that reads back as a path inside the
        # project, i.e. a different file than the one that was written.
        assert serialized == external.as_posix()

    def test_the_other_serializer_agrees(self) -> None:
        # The snapshot writer had its own copy of the rule, so it had its own
        # copy of the defect. Same input, same answer, one implementation.
        external = _sibling_of(PROJECT_ROOT.resolve()) / "sample-a.srt"

        assert artifacts_module._project_relative(external) == external.as_posix()

    def test_a_path_under_the_root_is_still_relative(self, tmp_path) -> None:
        inside = tmp_path / "tmp" / "sample-a.srt"

        assert (
            artifact_store.project_relative(inside, project_root=tmp_path)
            == "tmp/sample-a.srt"
        )

    def test_the_root_itself_is_the_current_directory(self, tmp_path) -> None:
        assert artifact_store.project_relative(tmp_path, project_root=tmp_path) == "."

    def test_a_string_that_merely_mentions_the_root_is_left_alone(self, tmp_path) -> None:
        # Substitution inside arbitrary strings is what made a sibling lose its
        # name. Log lines relativize their paths where they are formatted.
        line = f"wrote {tmp_path.as_posix()}/tmp/sample-a.srt and exited"

        assert artifact_store.project_relative(line, project_root=tmp_path) == line


class TestTwoWritersOfOneTargetDoNotCollide:
    """`<name>.<pid>.tmp` is shared by every writer in the process.

    Two threads writing one target - the same quality report from two jobs
    sharing a stem, a retry racing the run it replaced - staged into the same
    file. The second `replace()` moved it out from under the first, which then
    failed with FileNotFoundError while publishing a payload it had written in
    full.
    """

    def test_both_writers_finish(self, monkeypatch, tmp_path) -> None:
        target = tmp_path / "quality-report.json"
        staged = threading.Barrier(2, timeout=10)
        real_write_text = Path.write_text

        def write_text_then_wait(self, *args, **kwargs):
            written = real_write_text(self, *args, **kwargs)
            if self.name.endswith(".tmp"):
                # Hold both writers between staging and publishing: with one
                # shared staging name this is exactly the losing interleaving.
                staged.wait()
            return written

        monkeypatch.setattr(Path, "write_text", write_text_then_wait)
        failures: list[BaseException] = []

        def write(index: int) -> None:
            try:
                artifact_store.write_json_atomic(
                    target, {"writer": index}, project_root=tmp_path
                )
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(exc)

        threads = [threading.Thread(target=write, args=(index,)) for index in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert not [thread for thread in threads if thread.is_alive()]
        assert not failures, failures
        assert json.loads(target.read_text(encoding="utf-8"))["writer"] in (1, 2)
        assert not list(tmp_path.glob("*.tmp"))

    def test_the_staging_name_is_unique_per_call(self, tmp_path) -> None:
        target = tmp_path / "timings.json"

        assert artifact_store.temp_publish_path(
            target
        ) != artifact_store.temp_publish_path(target)

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path) -> None:
        target = tmp_path / "timings.json"

        with pytest.raises(TypeError):
            artifact_store.write_json_atomic(
                target, {"unserializable": object()}, project_root=tmp_path
            )

        assert not target.exists()
        assert not list(tmp_path.glob("*.tmp"))


class TestAGenerationIsPublishedOrNotAtAll:
    """`main` checked cancellation and the writer then opened the final path.

    Between those two moments nothing owned the file. A retry that reached the
    writer while the previous run was still inside it truncated a good subtitle
    and left a partial one where it had been, and the Web layer's
    `expected_run_id` guard - which protects the *record* of a job's artifacts -
    cannot see a file being written at all.
    """

    @staticmethod
    def _outputs(root: Path, *, run_id: str, sequence: int | None = None):
        return artifact_store.RunOutputs(
            lease=artifact_store.claim_output(
                root / "out" / "sample-a.srt", project_root=root,
                run_id=run_id, job_id="sample-job",
            ),
        )

    def test_nothing_reaches_the_target_before_publish(self, tmp_path) -> None:
        outputs = self._outputs(tmp_path, run_id="run-a")
        target = tmp_path / "out" / "sample-a.srt"

        outputs.stage(
            target, name="srt", write=lambda staged: staged.write_text("1", encoding="utf-8")
        )

        assert not target.exists()

        outputs.publish()
        assert not target.exists()
        artifact_store.export_output(outputs.lease)
        assert target.read_text(encoding="utf-8") == "1"

    def test_a_failure_halfway_leaves_the_previous_success_intact(self, tmp_path) -> None:
        target = tmp_path / "out" / "sample-a.srt"
        target.parent.mkdir(parents=True)
        target.write_text("the good one", encoding="utf-8")
        outputs = self._outputs(tmp_path, run_id="run-b")

        outputs.stage(
            target, name="srt", write=lambda staged: staged.write_text("half", encoding="utf-8")
        )
        with pytest.raises(TypeError):
            outputs.stage_json(
                tmp_path / "out" / "sample-a.bilingual.json",
                {"blocks": object()},
                name="bilingual_json",
            )

        # The run never got as far as publishing, so the file a previous run
        # produced is still the file that is there.
        assert target.read_text(encoding="utf-8") == "the good one"

    def test_an_overtaken_run_does_not_republish_over_the_newer_one(self, tmp_path) -> None:
        target = tmp_path / "out" / "sample-a.srt"
        slow = self._outputs(tmp_path, run_id="run-old", sequence=1)
        fast = self._outputs(tmp_path, run_id="run-new", sequence=2)
        slow.stage(
            target, name="srt", write=lambda staged: staged.write_text("old", encoding="utf-8")
        )
        fast.stage(
            target, name="srt", write=lambda staged: staged.write_text("new", encoding="utf-8")
        )

        # The retry finishes first; the run it replaced lands afterwards.
        fast.publish()
        artifact_store.export_output(fast.lease)
        with pytest.raises(artifact_store.OutputsSupersededError):
            slow.publish()

        assert target.read_text(encoding="utf-8") == "new"
        manifest = artifact_store.read_run_manifest(fast.lease)
        assert manifest["run_id"] == "run-new"

    def test_the_caller_can_veto_before_anything_moves(self, tmp_path) -> None:
        # The registry knows about a retry that has not written anything yet,
        # which no file on disk can show.
        outputs = self._outputs(tmp_path, run_id="run-old")
        target = tmp_path / "out" / "sample-a.srt"
        outputs.stage(
            target, name="srt", write=lambda staged: staged.write_text("old", encoding="utf-8")
        )

        with pytest.raises(artifact_store.OutputsSupersededError):
            outputs.publish(still_current=lambda: False)

        assert not target.exists()

    def test_the_manifest_names_the_whole_generation(self, tmp_path) -> None:
        outputs = self._outputs(tmp_path, run_id="run-a")
        outputs.stage(
            tmp_path / "out" / "sample-a.srt",
            name="srt",
            write=lambda staged: staged.write_text("1", encoding="utf-8"),
        )
        outputs.stage_json(
            tmp_path / "tmp" / "sample-a.bilingual.json", {"blocks": []}, name="bilingual_json"
        )

        manifest = outputs.publish()

        assert set(manifest["files"]) == {"srt", "bilingual_json"}
        assert manifest["files"]["srt"].startswith("tmp/outputs/")
        assert (tmp_path / manifest["files"]["srt"]).read_text("utf-8") == "1"
        assert manifest["run_id"] == "run-a"

    def test_staging_is_cleaned_up_either_way(self, tmp_path) -> None:
        published = self._outputs(tmp_path, run_id="run-a")
        published.stage(
            tmp_path / "out" / "sample-a.srt",
            name="srt",
            write=lambda staged: staged.write_text("1", encoding="utf-8"),
        )
        published.publish()

        vetoed = self._outputs(tmp_path, run_id="run-b")
        vetoed.stage(
            tmp_path / "out" / "sample-a.srt",
            name="srt",
            write=lambda staged: staged.write_text("2", encoding="utf-8"),
        )
        with pytest.raises(artifact_store.OutputsSupersededError):
            vetoed.publish(still_current=lambda: False)

        assert not published.staging_dir.exists()
        assert not vetoed.staging_dir.exists()


class TestOpeningAndDownloadingShareOneRule:
    def test_a_path_outside_every_allowed_root_is_refused(self, tmp_path) -> None:
        outside = tmp_path / "elsewhere" / "sample-a.srt"
        outside.parent.mkdir(parents=True)
        outside.write_text("", encoding="utf-8")
        allowed = tmp_path / "out"
        allowed.mkdir()

        assert artifact_store.authorized_path([outside], [allowed]) is None

    def test_traversal_is_resolved_before_it_is_compared(self, tmp_path) -> None:
        allowed = tmp_path / "out"
        allowed.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("", encoding="utf-8")

        assert artifact_store.authorized_path([allowed / ".." / "secret.txt"], [allowed]) is None

    def test_the_first_allowed_candidate_wins(self, tmp_path) -> None:
        allowed = tmp_path / "out"
        allowed.mkdir()
        real = allowed / "sample-a.srt"
        real.write_text("", encoding="utf-8")

        assert artifact_store.authorized_path(
            [tmp_path / "missing.srt", real], [allowed]
        ) == real.resolve()
