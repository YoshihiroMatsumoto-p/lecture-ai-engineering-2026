"""講義ページを埋め込み、検索用SQLite DBを作成する。"""
import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite_vec
from openai import APIConnectionError, APIStatusError
from app import retrieval as r
from app.config import BUNDLED_RAG_DB, api_error_message


def embedding_text(page: dict) -> str:
    """ページ本文に講義・章・タイトルを付け、索引用の文章にする。"""
    return (f"第{page['lecture']}回 {page['document_title']}\n{page['section']}\n"
            f"{page['title']}\n{page['content']}")


def build_index(source: Path = r.RAG_SOURCE, destination: Path = r.RAG_DB,
                *, rebuild: bool = False, embedder=None) -> dict:
    """1ページ1行で保存する。既存DBの再作成は--rebuildで明示する。"""
    if destination.resolve() == BUNDLED_RAG_DB.resolve():
        raise ValueError("配布元のGemini用DBは上書きしません。.cache/rag/配下の別DBを指定してください。")
    if destination.exists() and not rebuild:
        raise FileExistsError("検索DBは既にあります。再作成する場合は--rebuildを指定してください。")
    pages = r.load_pages(source)
    if not pages:
        raise ValueError("対象の講義ページがありません。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".rag-build-", suffix=".sqlite3", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        texts = [embedding_text(page) for page in pages]
        print(f"[embedding] {len(pages)}ページ / {r.EMBEDDING_MODEL}", flush=True)
        vectors = (r.embed_texts(texts, "RETRIEVAL_DOCUMENT") if embedder is None
                   else [embedder(text, "RETRIEVAL_DOCUMENT") for text in texts])
        if len(vectors) != len(pages) or any(len(vector) != r.EMBEDDING_DIMENSIONS for vector in vectors):
            raise ValueError("埋め込みの件数・次元が設定と一致しません。")
        db = sqlite3.connect(temporary)
        try:
            db.execute("CREATE TABLE embedding_config (provider TEXT, model TEXT, dimensions INTEGER)")
            db.execute("INSERT INTO embedding_config VALUES (?, ?, ?)",
                       (r.EMBEDDING_PROVIDER, r.EMBEDDING_MODEL, r.EMBEDDING_DIMENSIONS))
            db.execute("""CREATE TABLE chunks (
                page_id TEXT PRIMARY KEY, document_id TEXT, lecture INTEGER,
                document_title TEXT, section TEXT, title TEXT,
                page INTEGER, content TEXT, embedding BLOB)""")
            for page, vector in zip(pages, vectors, strict=True):
                values = [page[key] for key in ["page_id", "document_id", "lecture", "document_title",
                                                "section", "title", "page", "content"]]
                db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           values + [sqlite_vec.serialize_float32(vector)])
            db.commit()
        finally:
            db.close()
        if temporary.stat().st_size > 10_000_000:
            raise ValueError("検索DBが保存上限の10MBを超えました。")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"pages": len(pages), "provider": r.EMBEDDING_PROVIDER,
            "model": r.EMBEDDING_MODEL, "dimensions": r.EMBEDDING_DIMENSIONS,
            "bytes": destination.stat().st_size, "database": str(destination)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=r.RAG_SOURCE)
    parser.add_argument("--db", type=Path, default=r.RAG_DB)
    parser.add_argument("--rebuild", action="store_true", help="全ページを再度埋め込んでDBを作り直す")
    args = parser.parse_args()
    try:
        result = build_index(args.source, args.db, rebuild=args.rebuild)
    except APIStatusError as error:
        raise SystemExit(api_error_message(error.status_code)) from None
    except APIConnectionError:
        raise SystemExit("OpenAIの埋め込みAPIに接続できませんでした。") from None
    except (ValueError, OSError, sqlite3.Error) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
