import json

import pytest

from tools.audits import semantic_blind_review as review
from tools.audits.translation_quality_cases import build_cases, plan_source_cases


def _report(tmp_path, name, arms):
    cases = [
        case for case in plan_source_cases(build_cases())
        if case["case_id"] in {"split-negation", "ordinary-greeting"}
    ]
    path = tmp_path / name / "report.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"synthetic_only": True, "cases": cases, "arms": arms}, ensure_ascii=False),
                    encoding="utf-8")
    return path, cases


def test_the_sheet_hides_arms_and_the_score_separates_checkpoint_from_ordinary_errors(tmp_path):
    base = ["不愿意一起来吗？", "并不是不想去。", "只是今天有点累。", "早上好。", "今天天气真好。", "一起走到车站好吗？", "好，走吧。"]
    worse = list(base)
    worse[1] = "我不想去。"          # the checkpoint: double negation lost
    worse[4] = "今天天气真糟。"      # an ordinary line made wrong
    path, cases = _report(tmp_path, "run", {})
    assert len(cases) == len(base)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["arms"] = {
        "baseline": {"status": "complete", "texts": base, "remaining_kana_or_empty_ids": [],
                     "timings": [{"batch_index": i, "request_count": 1} for i in range(len(base))]},
        "candidate": {"status": "complete", "texts": worse, "remaining_kana_or_empty_ids": [0],
                      "timings": [{"batch_index": i, "request_count": 2 if i == 1 else 1} for i in range(len(base))]},
    }
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "blind"

    summary = review.make([str(path)], out)

    sheet = (out / "sheet.md").read_text(encoding="utf-8")
    assert "baseline" not in sheet and "candidate" not in sheet
    assert "★ 检查点：并不是不想去" in sheet
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert set(labels.values()) == {None}
    with pytest.raises(ValueError):
        review.score(out)

    # The judge labels from the sheet: the two wrong renderings are 0.
    wrong = {"我不想去。", "今天天气真糟。"}
    for line in sheet.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[0] in labels:
            labels[parts[0]] = 0 if parts[1] in wrong else 1
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    result = review.score(out)

    assert summary["arms"] == ["baseline", "candidate"]
    assert result["baseline"]["checkpoint_errors"] == 0 and result["baseline"]["ordinary_errors"] == 0
    assert result["candidate"]["checkpoint_errors"] == 1
    assert result["candidate"]["ordinary_errors"] == 1
    assert result["candidate"]["character_residue"] == 1
    assert result["candidate"]["extra_requests"] == 1
    assert result["baseline"]["units"] == result["candidate"]["units"] == len(base)


def test_it_refuses_to_overwrite_a_labelled_review(tmp_path):
    path, cases = _report(tmp_path, "run", {"a": {"status": "complete", "texts": ["x"] * 7}})
    out = tmp_path / "blind"
    review.make([str(path)], out)
    with pytest.raises(ValueError):
        review.make([str(path)], out)


def test_duplicate_arm_names_cannot_silently_replace_a_comparison(tmp_path):
    path, _ = _report(tmp_path, "run", {"candidate": {"status": "complete", "texts": ["译文"] * 7}})
    out = tmp_path / "blind"
    with pytest.raises(ValueError, match="duplicate arm"):
        review.make([f"same={path}", f"same={path}"], out)
    assert not out.exists()


def test_too_many_distinct_outputs_cannot_be_silently_omitted(tmp_path):
    path, _ = _report(tmp_path, "run", {
        f"arm-{i}": {"status": "complete", "texts": [f"译文{i}"] * 7} for i in range(27)
    })
    out = tmp_path / "blind"
    with pytest.raises(ValueError, match="distinct outputs"):
        review.make([str(path)], out)
    assert not out.exists()
