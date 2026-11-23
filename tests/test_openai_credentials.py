"""環境変数のない再起動後の読み込みと、秘密値を表示しない失敗処理を確認する。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.models.openai import OpenAIResponsesModel

from app import config


def test_environment_key_takes_priority_without_accessing_keychain(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-key")
    reader = Mock(side_effect=AssertionError("キーチェーンを読む必要はない"))
    monkeypatch.setattr(config.subprocess, "run", reader)
    assert config.get_openai_api_key() == "test-environment-key"
    reader.assert_not_called()


@pytest.mark.parametrize("module_name", ["agent", "judge"])
def test_openai_clients_work_without_shell_key_using_keychain(monkeypatch, capsys, module_name):
    from app import agent, judge

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.getpass, "getuser", lambda: "test-user")
    reader = Mock(return_value=SimpleNamespace(returncode=0, stdout="test-keychain-key\n"))
    monkeypatch.setattr(config.subprocess, "run", reader)
    model = agent.build_model() if module_name == "agent" else judge.build_llm_model(judge.DEFAULT_MODEL)
    assert isinstance(model, OpenAIResponsesModel)
    assert reader.call_args.args[0] == [
        "/usr/bin/security", "find-generic-password", "-a", "test-user",
        "-s", "aie2026-openai-api-key", "-w",
    ]
    assert reader.call_args.kwargs["capture_output"] is True
    assert "test-keychain-key" not in str(reader.call_args)
    captured = capsys.readouterr()
    assert "test-keychain-key" not in captured.out + captured.err


@pytest.mark.parametrize("returncode, message", [(44, "未登録"), (1, "許可されませんでした")])
def test_keychain_failures_do_not_disclose_output(monkeypatch, returncode, message):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=returncode, stdout="test-private-output", stderr="test-private-error",
    )))
    with pytest.raises(SystemExit, match=message) as exc:
        config.get_openai_api_key()
    assert "test-private" not in str(exc.value)


def test_non_mac_uses_environment_setup_instructions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config.sys, "platform", "linux")
    reader = Mock()
    monkeypatch.setattr(config.subprocess, "run", reader)
    with pytest.raises(SystemExit, match="環境変数 OPENAI_API_KEY"):
        config.get_openai_api_key()
    reader.assert_not_called()
