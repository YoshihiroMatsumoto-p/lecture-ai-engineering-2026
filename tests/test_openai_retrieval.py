"""OpenAIへの埋め込み切替、モデル混在防止、APIエラーの表示を確認する。"""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from app import retrieval, tools
from app.config import BUNDLED_RAG_DB


@pytest.fixture
def embedding_client(monkeypatch):
    # トークナイザの初回ダウンロードもAPIキーも、単体テストでは不要。
    encoding = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    monkeypatch.setattr(retrieval.tiktoken, "encoding_for_model", lambda _: encoding)
    monkeypatch.setattr(retrieval, "get_openai_api_key", lambda: "test-key")
    monkeypatch.setattr(retrieval, "EMBEDDING_DIMENSIONS", 2)
    client = Mock()
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=client)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(retrieval, "OpenAI", factory)
    return factory, client


def test_documents_and_queries_use_same_model_and_preserve_batch_order(embedding_client):
    factory, client = embedding_client

    def response(**kwargs):
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["dimensions"] == 2
        assert kwargs["encoding_format"] == "float"
        # 逆順のレスポンスでも、各入力のベクトルが入れ替わらないこと。
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[tokens[0], 1])
            for i, tokens in reversed(list(enumerate(kwargs["input"])))
        ])

    client.embeddings.create.side_effect = response
    texts = [chr(65 + i % 26) for i in range(33)]
    vectors = retrieval.embed_texts(texts, "RETRIEVAL_DOCUMENT")
    query = retrieval.embed_text("A", "RETRIEVAL_QUERY")
    assert vectors[0] == query
    assert vectors == [pytest.approx(retrieval.normalized_vector([ord(text), 1])) for text in texts]
    assert [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list] == [32, 1, 1]
    assert factory.call_args.kwargs["api_key"] == "test-key"


@pytest.mark.parametrize("text", ["", "  ", "x" * 8192])
def test_invalid_input_stops_before_api_call(embedding_client, text):
    factory, _ = embedding_client
    with pytest.raises(ValueError):
        retrieval.embed_text(text, "RETRIEVAL_QUERY")
    factory.assert_not_called()


@pytest.mark.parametrize("data", [
    [],
    [SimpleNamespace(index=1, embedding=[1, 0])],
    [SimpleNamespace(index=0, embedding=[1])],
    [SimpleNamespace(index=0, embedding=[float("nan"), 1])],
])
def test_invalid_embedding_response_is_rejected(embedding_client, data):
    _, client = embedding_client
    client.embeddings.create.return_value = SimpleNamespace(data=data)
    with pytest.raises(ValueError):
        retrieval.embed_text("LoRA", "RETRIEVAL_QUERY")


def test_gemini_bundle_is_rejected_before_query_embedding(monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_DB", BUNDLED_RAG_DB)
    embed = Mock(side_effect=AssertionError("異なるモデルのDBではAPIを呼ばない"))
    monkeypatch.setattr(retrieval, "embed_text", embed)
    with pytest.raises(ValueError, match="OpenAI"):
        retrieval.search("LoRA")
    embed.assert_not_called()


@pytest.mark.parametrize("settings", [
    ("google", "gemini-embedding-001", 768),
    ("openai", "text-embedding-3-large", 768),
    ("openai", "text-embedding-3-small", 1536),
])
def test_index_model_mismatch_stops_before_api_call(tmp_path, monkeypatch, settings):
    path = tmp_path / "different.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE embedding_config (provider TEXT, model TEXT, dimensions INTEGER)")
        db.execute("INSERT INTO embedding_config VALUES (?, ?, ?)", settings)
    monkeypatch.setattr(retrieval, "RAG_DB", path)
    embed = Mock()
    monkeypatch.setattr(retrieval, "embed_text", embed)
    with pytest.raises(ValueError, match="一致しません"):
        retrieval.search("LoRA")
    embed.assert_not_called()


@pytest.mark.parametrize("tool_name, backend", [
    ("search_materials", "search"), ("search_graph", "search_graph"),
])
@pytest.mark.parametrize("status", [401, 429, None])
def test_tools_handle_openai_errors_without_exposing_details(monkeypatch, tool_name, backend, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    if status is None:
        error = APIConnectionError(request=request)
    else:
        error = APIStatusError("test-private-detail", response=httpx.Response(status, request=request),
                               body={"message": "test-private-detail"})
    monkeypatch.setattr(retrieval, backend, Mock(side_effect=error))
    result = json.loads(getattr(tools, tool_name)("LoRA"))
    assert result["error"]
    assert "test-private-detail" not in result["error"]
    if status == 429:
        assert "レート制限" in result["error"]
    elif status == 401:
        assert "401" in result["error"]
    else:
        assert "接続" in result["error"]
