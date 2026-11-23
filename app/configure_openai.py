"""OpenAI APIキーを、長さを変えずにMacのキーチェーンへ登録する。

uv run python -m app.configure_openai
キーは非表示の入力欄で受け取り、コマンド引数・ファイル・ログには残さない。
"""

import getpass
import hmac
import subprocess
import sys
import warnings

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

from app.config import DEFAULT_OPENAI_MODEL, OPENAI_KEYCHAIN_SERVICE


def _quote(value: str) -> str:
    """securityの対話コマンド用。改行を拒否し、引用符とバックスラッシュをエスケープ。"""
    if any(c in value for c in "\r\n\0"):
        raise ValueError("キーチェーンの項目名に不正な文字があります。")
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def save_key(key: str, *, service: str = OPENAI_KEYCHAIN_SERVICE) -> int:
    """キーを標準入力でsecurityに渡し、保存された全文と照合する。"""
    if sys.platform != "darwin":
        raise ValueError("この登録コマンドはMac専用です。")
    if not key.startswith("sk-") or not key.isascii() or any(c.isspace() for c in key) or "*" in key:
        raise ValueError("sk-から始まるAPIキーの全文を、空白や伏字を含めずに入力してください。")
    account = getpass.getuser()
    # -wのパスワード入力はlibc getpassの128文字制限に当たる。
    # security -iへ-X(16進表現)を標準入力で渡し、その経路を避ける。
    command = f"add-generic-password -U -a {_quote(account)} -s {_quote(service)} -X {key.encode('ascii').hex()}\n"
    if len(command.encode("utf-8")) >= 4096:
        raise ValueError("入力がsecurityのコマンド長上限を超えます。保存は行いません。")
    result = subprocess.run(
        ["/usr/bin/security", "-i"], input=command,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError("キーチェーンへの保存に失敗しました。Macのロックとアクセス許可を確認してください。")
    saved = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True, text=True, timeout=30,
    )
    if saved.returncode or not hmac.compare_digest(saved.stdout.removesuffix("\n"), key):
        raise RuntimeError("保存内容が入力したキーと一致しません。登録完了とは扱えません。")
    return len(key)


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Cursorなどのターミナルで実行してください。キーの入力は画面に表示されません。")
    try:
        # Pythonのgetpassはlibc getpassを使わず、128文字で切り捨てない。
        # 非表示入力に失敗した場合は、平文表示に切り替えず停止する。
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("OpenAI APIキーの全文を貼り付けてEnter: ").strip()
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        raise SystemExit("\n入力を中止しました。キーは保存していません。") from None
    try:
        length = save_key(key)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
    except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
        raise SystemExit("保存・照合を完了できませんでした。キーチェーンの状態を確認して再実行してください。") from None
    print(f"{length}文字を保存し、入力したキーとの完全一致を確認しました。")
    try:
        with OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=15) as client:
            client.models.retrieve(DEFAULT_OPENAI_MODEL)
    except AuthenticationError:
        raise SystemExit("全文の保存は成功しましたが、OpenAI認証は401でした。コピー元のキーの有効性を確認してください。") from None
    except APIStatusError as error:
        raise SystemExit(f"モデル参照の確認はHTTP {error.status_code}で終了しました。権限などを確認してください。") from None
    except APIConnectionError:
        raise SystemExit("全文の保存は成功しましたが、OpenAIへの接続確認ができませんでした。") from None
    print("OpenAI認証も成功しました。次の1行で採点できます。")
    print("env -u OPENAI_API_KEY uv run python -m app.judge")


if __name__ == "__main__":
    main()
