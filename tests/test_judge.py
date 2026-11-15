"""第4回: 採点（app/judge.py）の配線テスト。API キーは使わない。

--model test と同じ偽モデル（規則で判定する FunctionModel）で、評価セット 10 件の
「採点 → 人のラベルとの比較 → 要確認・不一致の集計」が通ることを確かめる。
"""

import json

import pytest
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel

from app import judge
from app.judge import (
    ALL_CRITERIA,
    DEFAULT_CASES,
    DEFAULT_RUBRIC,
    build_judge,
    build_test_model,
    check_lecture_mark,
    judge_cases,
    load_labels,
    overall,
    read_jsonl,
    summarize,
)

INSTRUCTOR_LABELS = judge.ROOT / "lectures" / "04" / "reference" / "labels_instructor.jsonl"


def test_lecture_mark_is_checked_by_code():
    assert check_lecture_mark("LoRA は…\n\n関係する回：第2回") == "pass"
    assert check_lecture_mark("第2回で扱う LoRA は…\n\n以上です。") == "fail"  # 末尾に無ければ fail
    assert check_lecture_mark("") == "fail"


def test_overall_requires_all_required_criteria():
    base = {"grounding": "pass", "scope": "pass", "hint": "pass", "lecture_mark": "pass", "clarity": "fail"}
    assert overall(base) == "pass"  # 分かりやすさは総合に入れない
    assert overall({**base, "hint": "fail"}) == "fail"
    assert overall({**base, "grounding": "unsure"}) == "unsure"


def test_cases_and_rubric_are_consistent():
    cases = read_jsonl(DEFAULT_CASES)
    assert len(cases) == 10
    assert len({c["id"] for c in cases}) == 10
    for c in cases:
        assert c["question"] and c["answer"] and c["reference"]["expected"]
    rubric = DEFAULT_RUBRIC.read_text(encoding="utf-8")
    for key, name in ALL_CRITERIA[:-1]:  # 回の表示はコードで判定するので rubric には書かない
        assert f"{name}（{key}）" in rubric
    labels = load_labels(INSTRUCTOR_LABELS)
    assert set(labels) == {c["id"] for c in cases}


def test_judge_pipeline_with_fake_model():
    cases = read_jsonl(DEFAULT_CASES)
    labels = load_labels(INSTRUCTOR_LABELS)
    agent = build_judge(build_test_model(), DEFAULT_RUBRIC.read_text(encoding="utf-8"))

    verdicts = judge_cases(agent, cases)
    assert set(verdicts) == {c["id"] for c in cases}
    for v in verdicts.values():
        for key, _ in ALL_CRITERIA:
            assert v[key]["verdict"] in ("pass", "fail", "unsure")
            assert v[key]["reason"]

    summary = summarize(cases, verdicts, labels)
    assert summary["n"] == 10
    # unsure にした項目は一致率の分母から外れ、要確認に入る
    excluded = sum(p["total"] - p["compared"] for p in summary["per_criterion"].values())
    assert excluded == len(summary["review"])
    assert summary["coverage"]["covered"] + len({r["id"] for r in summary["review"]}) == 10
    for m in summary["mismatches"]:
        assert m["kind"] in ("見逃し", "過検出")


@pytest.mark.parametrize("name", [
    "gpt-4.1-mini-2025-04-14",
    "openai:gpt-4.1-mini-2025-04-14",
    "openai-responses:gpt-4.1-mini-2025-04-14",
])
def test_openai_judge_does_not_require_gemini_key(monkeypatch, name):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = judge.build_llm_model(name)
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "gpt-4.1-mini-2025-04-14"


def test_gemini_judge_remains_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(judge.build_llm_model("gemini-3.5-flash-lite"), GoogleModel)


def test_generated_answers_include_observed_tool_calls(monkeypatch):
    import app.agent as agent_module

    model = TestModel(call_tools=["search_glossary"], custom_output_text="ツールの実行はプログラムが担当します。\n第3回")
    monkeypatch.setattr(agent_module, "build_model", lambda: model)
    cases = [{"id": "01", "question": "ツールとは？", "answer": ""}]
    result = judge.generate_answers(cases, wait=0)
    assert result[0]["answer"].endswith("第3回")
    generated = result[0]["generation"]
    assert [call["name"] for call in generated["tool_calls"]] == ["search_glossary"]
    assert generated["usage"]["requests"] == 2
    assert generated["usage"]["tool_calls"] == 1


def test_generation_does_not_approve_writes_or_treat_deferral_as_answer(monkeypatch, tmp_path):
    import app.agent as agent_module

    monkeypatch.setattr(agent_module, "build_model", lambda: TestModel(call_tools=["save_learning_log"]))
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(judge.tools, "LEARNING_LOG_DIR", log_dir)
    cases = [{"id": "01", "question": "学習内容を保存して", "answer": ""}]
    with pytest.raises(SystemExit, match="承認待ち"):
        judge.generate_answers(cases, wait=0)
    assert cases[0]["answer"] == ""
    assert not log_dir.exists()


def test_cli_does_not_present_ai_draft_labels_as_human_labels(tmp_path, capsys):
    labels_path = tmp_path / "labels.jsonl"
    out_path = tmp_path / "result.json"
    rows = read_jsonl(INSTRUCTOR_LABELS)
    for row in rows:
        row.update(label_source="assistant_draft", human_reviewed=False)
    judge.write_jsonl(labels_path, rows)
    judge.main(["--model", "test", "--labels", str(labels_path), "--out", str(out_path)])
    result = json.loads(out_path.read_text())
    assert result["labeler"] == "AIが作成した仮ラベル（本人未確認）"
    assert result["labeler"] in capsys.readouterr().out
