from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scorer_v11_retrain_driver_preserves_execution_contracts() -> None:
    script = (
        PROJECT_ROOT
        / "tools"
        / "boundary"
        / "ja"
        / "run_candidate_island_scorer_v11_retrain.ps1"
    ).read_text(encoding="utf-8")

    assert "$env:PYTHONIOENCODING = 'utf-8'" in script
    assert "$env:UV_CACHE_DIR" in script
    assert "uv run python" in script
    assert "compile_candidate_island_scorer_v11_real_train_manual.py" in script
    assert "rebind_candidate_island_scorer_v11_raw_features.py" in script
    assert "extract_candidate_island_scorer_v11_raw_features.py" in script
    assert "train_candidate_island_scorer_v11.py" in script
    assert "score_candidate_island_scorer_v11_checkpoint.py" in script
    assert "generate_candidate_island_scorer_v11_prediction_audit_html.py" in script
    assert "source_predictions.jsonl" in script
    assert "--smoke" in script
    assert "--early-stopping-patience" in script
    assert "full_p2048_h256" in script
    assert "--device', 'cuda'" in script
    assert "[ValidateSet('prepare', 'extract', 'features', 'smoke', 'full', 'gate', 'all')]" in script
    assert "http://127.0.0.1:8080/agents/audits/" in script
    assert "0.6B" not in script
    assert "conda" not in script.lower()
    assert "wsl" not in script.lower()
