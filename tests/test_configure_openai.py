"""長いキーを切り捨てず、プロセス引数へ秘密を渡さない登録処理。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import configure_openai as setup


def test_long_key_is_passed_via_stdin_and_verified(monkeypatch, capsys):
    key = "sk-proj-" + "x" * 192
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    runner = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout=key + "\n", stderr=""),
    ])
    monkeypatch.setattr(setup.subprocess, "run", runner)
    assert setup.save_key(key) == 200
    write, read = runner.call_args_list
    assert write.args[0] == ["/usr/bin/security", "-i"]
    assert key.encode().hex() in write.kwargs["input"]
    assert key not in str(write.args) + str(read.args)
    assert key.encode().hex() not in str(write.args) + str(read.args)
    assert write.kwargs["capture_output"] is True
    assert read.args[0][-1] == "-w"
    output = capsys.readouterr()
    assert key not in output.out + output.err


def test_truncated_saved_key_is_not_reported_as_success(monkeypatch):
    key = "sk-proj-" + "x" * 192
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    monkeypatch.setattr(setup.subprocess, "run", Mock(side_effect=[
        SimpleNamespace(returncode=0),
        SimpleNamespace(returncode=0, stdout=key[:128] + "\n"),
    ]))
    with pytest.raises(RuntimeError, match="一致しません"):
        setup.save_key(key)


@pytest.mark.parametrize("key", ["", "sk-proj-abc def", "sk-proj-***", "sk-proj-日本語", "sk-proj-" + "x" * 3000])
def test_invalid_or_oversize_input_does_not_modify_keychain(monkeypatch, key):
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    runner = Mock()
    monkeypatch.setattr(setup.subprocess, "run", runner)
    with pytest.raises(ValueError):
        setup.save_key(key)
    runner.assert_not_called()
