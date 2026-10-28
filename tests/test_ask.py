from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="回答",
    )

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = SimpleNamespace(generate_content=lambda **kwargs: response)

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "応答時間:" in output
    assert "入力トークン: 12" in output
    assert "出力トークン: 7" in output


def test_ask_prints_placeholder_when_usage_metadata_is_missing(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = SimpleNamespace(generate_content=lambda **kwargs: response)

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "入力トークン: -" in output
    assert "出力トークン: -" in output


def test_ask_interactively_keeps_history_between_questions(monkeypatch, capsys):
    responses = [
        SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
            text="最初の回答",
        ),
        SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=5),
            text="続きの回答",
        ),
    ]
    questions = []

    class FakeChat:
        def send_message(self, question):
            questions.append(question)
            return responses[len(questions) - 1]

    class FakeClient:
        def __init__(self, *, api_key):
            def create_chat(*, model):
                assert model == ask_module.MODEL
                return FakeChat()

            self.chats = SimpleNamespace(create=create_chat)

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    inputs = iter(["最初の質問", "その続き", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask_module.ask_interactively()

    output = capsys.readouterr().out
    assert questions == ["最初の質問", "その続き"]
    assert "最初の回答" in output
    assert "続きの回答" in output
    assert output.count("[usage]") == 2


def test_compare_runs_each_question_with_each_model(monkeypatch, capsys):
    calls = []

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, *, model, contents):
            calls.append((model, contents))
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1,
                    candidates_token_count=2,
                ),
                text=f"{model}への回答",
            )

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    ask_module.compare(["model-a", "model-b"], ["質問1", "質問2"])

    output = capsys.readouterr().out
    assert calls == [
        ("model-a", "質問1"),
        ("model-b", "質問1"),
        ("model-a", "質問2"),
        ("model-b", "質問2"),
    ]
    assert output.count("応答時間:") == 4
    assert output.count("入力トークン: 1") == 4
    assert output.count("出力トークン: 2") == 4