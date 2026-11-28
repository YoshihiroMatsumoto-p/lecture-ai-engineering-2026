"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from openai import OpenAI, APIConnectionError, APIStatusError
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec
import tiktoken

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_openai_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    api_error_message,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("埋め込みがゼロベクトル、または有限でない値です。")
    return [value / norm for value in values]


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """OpenAIで文章をまとめて埋め込む。本文と質問に同じモデル・次元を使う。"""
    # 呼び出し元の互換性のためtask_typeを受け取る。OpenAIへは渡さない。
    if task_type not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}:
        raise ValueError("埋め込みの用途が不正です。")
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("埋め込み入力が空です。")
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    inputs = [encoding.encode(text, disallowed_special=()) for text in texts]
    for tokens in inputs:
        if len(tokens) > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えます: {len(tokens)} tokens。本文は切り捨てません。")
    vectors = []
    with OpenAI(api_key=get_openai_api_key(), timeout=30.0, max_retries=2) as client:
        # 最大32×8,191 tokensで、1リクエストの総トークン上限にも収める。
        for start in range(0, len(inputs), 32):
            batch = inputs[start:start + 32]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch,
                dimensions=EMBEDDING_DIMENSIONS, encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("埋め込みAPIの結果件数・順序が入力と一致しません。")
            for item in ordered:
                if len(item.embedding) != EMBEDDING_DIMENSIONS:
                    raise ValueError("埋め込みの次元数が設定と一致しません。")
                vectors.append(normalized_vector(item.embedding))
    return vectors


def embed_text(text: str, task_type: str) -> list[float]:
    """1件の文章を埋め込む。検索の質問にも索引と同じOpenAIモデルを使う。"""
    return embed_texts([text], task_type)[0]


def check_index_model(db: sqlite3.Connection) -> None:
    """異なるモデルで作ったDBを使って検索しないよう、APIを呼ぶ前に照合する。"""
    try:
        settings = db.execute("SELECT provider, model, dimensions FROM embedding_config").fetchall()
    except sqlite3.OperationalError:
        raise ValueError(
            "このDBにはOpenAI用の埋め込み情報がありません。AIE_RAG_DBを確認し、"
            "uv run python -m scripts.build_rag_index でOpenAI用DBを作成してください。"
        ) from None
    expected = (EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    if len(settings) != 1 or tuple(settings[0]) != expected:
        raise ValueError("検索DBの埋め込みモデル・次元が現在の設定と一致しません。対応するDBを指定するか再作成してください。")


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("OpenAI用検索DBがありません。uv run python -m scripts.build_rag_index を先に実行してください。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        check_index_model(db)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        return [dict(row) for row in rows]


def search_with_neighbors(query: str) -> list[dict]:
    """検索で見つかった各ページに、同じ資料の前後1ページずつを追加して返す。"""
    hits = search(query)

    # 資料IDとページ番号の組み合わせで、本文や引用情報を取得できるようにする
    pages_by_location = {
        (page["document_id"], page["page"]): page
        for page in load_pages()
    }

    # 通常の検索結果を先に入れ、追加済みのページIDを記録する
    results = list(hits)
    seen = {page["page_id"] for page in hits}

    for hit in hits:
        for number in (hit["page"] - 1, hit["page"] + 1):
            neighbor = pages_by_location.get(
                (hit["document_id"], number)
            )

            # 前後のページが存在し、まだ結果に含まれていなければ追加する
            if neighbor is not None and neighbor["page_id"] not in seen:
                results.append(neighbor)
                seen.add(neighbor["page_id"])

    return results


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    try:
        results = search(args.query)
    except APIStatusError as error:
        raise SystemExit(api_error_message(error.status_code)) from None
    except APIConnectionError:
        raise SystemExit("OpenAIの埋め込みAPIに接続できませんでした。") from None
    except (ValueError, OSError, sqlite3.Error) as error:
        raise SystemExit(str(error)) from None
    for i, result in enumerate(results, 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
