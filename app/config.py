"""講座全体で共通の設定。

APIキーは環境変数、OpenAI用はMacのキーチェーンからも読み込む。
"""

import getpass
import os
import subprocess
import sys
from pathlib import Path

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   例: export AIE_MODEL=gemini-3.5-flash
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = os.getenv("AIE_MODEL", DEFAULT_MODEL)
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
OPENAI_MODEL = os.getenv("AIE_MODEL", DEFAULT_OPENAI_MODEL)
OPENAI_KEYCHAIN_SERVICE = "aie2026-openai-api-key"


def get_openai_api_key() -> str:
    """環境変数を優先し、未設定ならこの講座専用のMacキーチェーン項目を読む。"""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-a", getpass.getuser(),
                 "-s", OPENAI_KEYCHAIN_SERVICE, "-w"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise SystemExit("OpenAI APIキーをキーチェーンから読み取れませんでした。Macのロックを解除して再実行してください。") from None
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if result.returncode not in (0, 44):  # 44: 指定した項目が存在しない
            raise SystemExit("OpenAI APIキーへのキーチェーンアクセスが許可されませんでした。キーチェーンのロックと許可画面を確認してください。")
        raise SystemExit(
            "OpenAI APIキーが未登録です。ターミナルで次を実行し、表示される入力欄にキーを貼り付けてください。\n"
            "uv run python -m app.configure_openai\n"
            "入力したキーは画面に表示されません。登録後は再起動しても自動で読み込めます。"
        )
    raise SystemExit("環境変数 OPENAI_API_KEY が見つかりません。実行環境のシークレットまたは環境変数に設定してください。")

# 第5回: 1ページを1チャンクとして検索する。索引はscripts/build_rag_index.pyで作成する。
ROOT = Path(__file__).resolve().parent.parent
RAG_SOURCE = ROOT / "data/rag_data/lectures.json"
RAG_TREE = ROOT / "data/rag_data/tree.json"
RAG_GRAPH = ROOT / "data/rag_data/graph.json"
BUNDLED_RAG_DB = ROOT / "data/rag_data/index.sqlite3"
RAG_DB = Path(os.getenv("AIE_RAG_DB", str(ROOT / ".cache/rag/openai.sqlite3")))
RAG_DOCUMENT_IDS = ("lec01-main", "lec02-main", "lec03-main")
RAG_TOP_K = 5
GRAPH_START_K = 3
GRAPH_HOPS = 2
GRAPH_MAX_RESULTS = 10
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 768
# OpenAIの入力上限に余裕を残す。超過時は本文を切り捨てず停止する。
EMBEDDING_MAX_TOKENS = 8191


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        return "レート制限に達しました。1分ほど待ってから再実行してください。"
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 GEMINI_API_KEY が見つかりません。\n"
            "1. GitHub の Settings → Codespaces → Secrets に GEMINI_API_KEY を登録し、\n"
            "   使う repository に自分の fork を選んでください。\n"
            "2. Codespaces を起動したあとに登録した場合は、Codespaces を再起動してください。\n"
            "   （左下の「Codespaces」→「Stop Current Codespace」→ 再度開く）"
        )
    return key
