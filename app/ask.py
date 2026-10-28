"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 会話履歴を保って続けて質問する
    uv run python -m app.ask --model gemini-3.5-flash-lite --model gemini-3.5-flash \\
        --question "LoRAとは何ですか？"

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentResponse

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, model: str = MODEL) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed_seconds = perf_counter() - started_at

    print_usage(response, elapsed_seconds)
    return response.text


def compare(models: list[str], questions: list[str]) -> None:
    for question in questions:
        for model in models:
            print(f"[model] {model}")
            print(f"[質問] {question}\n")
            print(ask(question, model))
            print()


def ask_interactively() -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print("終了するには exit、quit、または 終了 と入力してください。")

    while True:
        try:
            question = input("[質問] ").strip()
        except EOFError:
            print()
            return

        if question.casefold() in {"exit", "quit", "終了"}:
            return
        if not question:
            continue

        started_at = perf_counter()
        response = chat.send_message(question)
        elapsed_seconds = perf_counter() - started_at
        print_usage(response, elapsed_seconds)
        print(response.text)


def print_usage(response: GenerateContentResponse, elapsed_seconds: float) -> None:
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    print(
        f"[usage] 応答時間: {elapsed_seconds:.3f}秒, "
        f"入力トークン: {input_tokens if input_tokens is not None else '-'}, "
        f"出力トークン: {output_tokens if output_tokens is not None else '-'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="会話履歴を保って続けて質問する",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="比較するモデル（複数回指定可。既定値: AIE_MODEL または標準モデル）",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        metavar="QUESTION",
        help="比較する質問（複数回指定可）",
    )
    args = parser.parse_args()

    if args.interactive and (
        args.question is not None or args.models is not None or args.questions is not None
    ):
        parser.error("--interactive と質問・モデルの指定は同時に使えません")
    if args.question is not None and args.questions is not None:
        parser.error("位置引数の質問と --question は同時に指定できません")

    try:
        if args.interactive:
            print(f"[model] {MODEL}")
            ask_interactively()
        else:
            models = args.models or [MODEL]
            questions = args.questions or (
                [args.question] if args.question is not None else [DEFAULT_QUESTION]
            )
            compare(models, questions)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
