"""What the offline alignment tools are allowed to load, and where they run.

Two things used to be decided silently. `torch.load(..., weights_only=False)`
executes whatever a checkpoint archive contains, before any tensor is read, and
these scripts take the path from the command line. And
`"cuda" if torch.cuda.is_available() else "cpu"` turns a missing driver into a
run that is a hundred times slower and looks like a success at the end of it.

Neither is wrong to want - a file you produced, a smoke test on a laptop - but
both have to be asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.align import checkpoint_io


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)

    @staticmethod
    def device(name: str) -> str:
        return name


@pytest.fixture
def without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(available=False))


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(available=True))


class TestChoosingADevice:
    def test_a_missing_gpu_is_an_error_not_a_slower_run(self, without_cuda):
        with pytest.raises(SystemExit, match="CUDA"):
            checkpoint_io.resolve_device("auto")

    def test_cpu_is_available_to_whoever_asks_for_it(self, without_cuda):
        assert checkpoint_io.resolve_device("cpu") == "cpu"
        assert checkpoint_io.resolve_device("auto", allow_cpu=True) == "cpu"

    def test_pinning_cuda_still_fails_when_there_is_none(self, without_cuda):
        # `--device cuda --allow-cpu` is a contradiction, and the explicit half
        # of it wins: the run asked for the GPU by name.
        with pytest.raises(SystemExit):
            checkpoint_io.resolve_device("cuda", allow_cpu=True)

    def test_a_present_gpu_is_used(self, with_cuda):
        assert checkpoint_io.resolve_device("auto") == "cuda"
        assert checkpoint_io.resolve_device("cpu") == "cpu"


class TestLoadingACheckpoint:
    def test_a_plain_checkpoint_loads_under_the_strict_reader(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "head.pt"
        torch.save({"schema": "x", "weight": torch.zeros(2)}, path)

        payload = checkpoint_io.load_checkpoint(path)

        assert payload["schema"] == "x"

    def test_a_checkpoint_carrying_objects_is_refused_by_default(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "pickled.pt"
        torch.save({"vocab": argparse.Namespace(chars="abc")}, path)

        with pytest.raises(SystemExit, match="--trust-checkpoint"):
            checkpoint_io.load_checkpoint(path)

    def test_the_operator_can_vouch_for_their_own_file(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "pickled.pt"
        torch.save({"vocab": argparse.Namespace(chars="abc")}, path)

        payload = checkpoint_io.load_checkpoint(path, trusted=True)

        assert payload["vocab"].chars == "abc"


def test_the_production_alignment_head_reads_without_the_permissive_loader():
    """The one checkpoint a user does not necessarily produce themselves.

    `ASR_ALIGNMENT_HEAD_PATH` accepts `hf:<repo>@<sha>#<file>`, so the file can
    arrive over the network; the loader there is strict for that reason. This
    holds the shipped head to what that loader accepts.
    """
    torch = pytest.importorskip("torch")
    head_path = ROOT / "models" / "ctc_aligner_jav_vocalisation_v3.pt"
    if not head_path.exists():
        pytest.skip("the default alignment head is not present in this workspace")

    payload = torch.load(head_path, map_location="cpu", weights_only=True)

    assert payload["schema"]
