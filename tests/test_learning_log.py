"""学習記録と承認フローのテスト。TestModelを使い、実APIや個人の記録には触れない。"""

import json
import sys
from datetime import datetime, timedelta

import pytest
from pydantic_ai import DeferredToolRequests, models
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import tools

CONTENT = "ツールを選ぶのはLLMで、実行するのはPydantic AIだと学んだ。"


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", path)
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    return path


@pytest.fixture
def log_model(monkeypatch):
    model = TestModel(call_tools=["save_learning_log"])
    # TestModelの呼び出し引数を固定して、日本語の保存内容が引き継がれることを確かめる。
    monkeypatch.setattr(model, "gen_tool_args", lambda tool: {"lecture": 3, "content": CONTENT})
    return model


def test_save_learning_log_keeps_existing_records(log_dir):
    first_result = tools.save_learning_log(3, CONTENT)
    first_path = next(log_dir.glob("*.json"))
    first_bytes = first_path.read_bytes()
    tools.save_learning_log(2, "LoRAのランクについて復習した。")

    assert len(list(log_dir.glob("*.json"))) == 2
    assert first_path.read_bytes() == first_bytes
    assert str(first_path) in first_result
    record = json.loads(first_path.read_text(encoding="utf-8"))
    assert record["lecture"] == 3
    assert record["content"] == CONTENT
    assert datetime.fromisoformat(record["saved_at"]).utcoffset() == timedelta(hours=9)


@pytest.mark.parametrize("lecture, content", [(0, CONTENT), (-1, CONTENT), (3, " \n　")])
def test_invalid_learning_log_does_not_create_directory(log_dir, lecture, content):
    assert "保存できません" in tools.save_learning_log(lecture, content)
    assert not log_dir.exists()


def test_learning_log_waits_for_approval_without_writing(log_dir, log_model):
    result = agent_module.agent.run_sync("第3回の学習内容を保存して", model=log_model)

    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    call = result.output.approvals[0]
    assert call.tool_name == "save_learning_log"
    assert call.args_as_dict() == {"lecture": 3, "content": CONTENT}
    assert not log_dir.exists()


@pytest.mark.parametrize("reply, approved", [("y", True), ("YES", True), ("n", False), ("", False)])
def test_cli_previews_content_and_saves_only_when_approved(
    log_dir, log_model, monkeypatch, capsys, reply, approved
):
    def answer(prompt):
        preview = capsys.readouterr().out
        assert CONTENT in preview
        assert str(log_dir) in preview
        assert "[y/N]" in prompt
        assert not log_dir.exists()
        return reply

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(agent_module, "build_model", lambda: log_model)
    monkeypatch.setattr(sys, "argv", ["app.agent", "--trace", "第3回の学習内容を保存して"])
    agent_module.main()

    output = capsys.readouterr().out
    assert "[tool call]   save_learning_log" in output
    assert "requests=2" in output  # 承認前と再開後のLLM呼び出しを合算する。
    if approved:
        paths = list(log_dir.glob("*.json"))
        assert len(paths) == 1
        assert json.loads(paths[0].read_text(encoding="utf-8"))["content"] == CONTENT
        assert "学習記録を保存しました" in output
    else:
        assert not log_dir.exists()
        assert "拒否しました" in output


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_interrupted_approval_does_not_save(log_dir, log_model, monkeypatch, interruption):
    def interrupt(prompt):
        raise interruption()

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(SystemExit) as exc:
        agent_module.run_with_approval(agent_module.agent, "学習内容を保存して", log_model)
    assert exc.value.code == 0
    assert not log_dir.exists()


def test_multiple_learning_logs_are_approved_individually(log_dir, log_model, monkeypatch):
    log_model.call_tools = ["save_learning_log", "save_learning_log"]
    arguments = iter([
        {"lecture": 3, "content": CONTENT},
        {"lecture": 2, "content": "この記録は拒否する。"},
    ])
    replies = iter(["y", "n"])
    monkeypatch.setattr(log_model, "gen_tool_args", lambda tool: next(arguments))
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    result = agent_module.run_with_approval(agent_module.agent, "2件の学習内容を保存して", log_model)

    assert isinstance(result.output, str)
    paths = list(log_dir.glob("*.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text(encoding="utf-8"))["content"] == CONTENT
