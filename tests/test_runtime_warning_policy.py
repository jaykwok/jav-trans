from pathlib import Path


def test_runtime_entrypoints_do_not_suppress_all_warnings():
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/main.py", "src/asr/pipeline.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert 'warnings.filterwarnings("ignore")' not in source
