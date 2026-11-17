"""SciFact BM25の実行用CLI。引数、データ取得・正規化、結果・メタデータ保存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from bm25_baseline import Hit, configure_bm25, score_breakdown, search_query

if TYPE_CHECKING:
    from pyserini.search.lucene import LuceneSearcher

type Identifier = int | str
type Topics = Mapping[Identifier, Mapping[str, str]]
type RawQrels = Mapping[Identifier, Mapping[Identifier, str]]
type Queries = dict[str, str]
type Qrels = dict[str, dict[str, int]]

INDEX = "beir-v1.0.0-scifact.flat"
COLLECTION = "beir-v1.0.0-scifact-test"
INDEX_README = (
    "https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/blob/main/"
    "lucene-inverted/flat/lucene-inverted.beir-v1.0.0-flat.20221116.505594.README.md"
)


def normalize_data(topics: Topics, qrels: RawQrels) -> tuple[Queries, Qrels]:
    """PyseriniのtopicsとqrelsのIDを文字列、関連度を整数に揃える。

    topicsからはtitleをクエリ本文として取り出す。qrelsにあるクエリが
    topicsに存在しない場合はValueErrorとし、不完全な入力での実行を防ぐ。
    戻り値はクエリIDと本文の辞書、およびクエリID・文書ID・関連度の辞書。
    """
    queries = {str(qid): fields["title"] for qid, fields in topics.items()}
    judgments = {
        str(qid): {str(docid): int(grade) for docid, grade in docs.items()}
        for qid, docs in qrels.items()
    }
    missing = judgments.keys() - queries.keys()
    if missing:
        raise ValueError(f"qrelsに対応するtopicsがありません: {sorted(missing)}")
    return queries, judgments


def format_run(qid: str, hits: Sequence[Hit]) -> str:
    """順位順の検索結果を、1始まりの順位を持つTREC run形式に変換する。"""
    return "".join(
        f"{qid} Q0 {hit.docid} {rank} {hit.score:.17g} bm25-scifact\n"
        for rank, hit in enumerate(hits, 1)
    )


def runtime_metadata(
    searcher: LuceneSearcher, info: Mapping[str, object]
) -> dict[str, object]:
    """実行環境・索引仕様・検索条件を記録するメタデータを集める。

    searcherは読み込み済みの検索器、infoはPyseriniの索引カタログの項目。
    MD5はカタログにある配布アーカイブの公開値を記録する。
    """
    from pyserini.pyclass import autoclass

    system = autoclass("java.lang.System")
    lucene = autoclass("org.apache.lucene.util.Version")
    return {
        "python": platform.python_version(),
        "pyserini": version("pyserini"),
        "java": system.getProperty("java.version"),
        "lucene": lucene.LATEST.toString(),
        "anserini_jar": (
            searcher.object.getClass()
            .getProtectionDomain()
            .getCodeSource()
            .getLocation()
            .toString()
        ),
        "index": INDEX,
        "index_path": str(searcher.index_dir),
        "index_archive": info["filename"],
        "index_archive_md5": info["md5"],
        "index_checksum_source": "Pyserini catalog (published archive MD5)",
        "index_build_anserini_commit": "505594",
        "index_specification": INDEX_README,
        "index_stats": searcher.index_reader.stats(),
        "has_document_vectors": searcher.index_reader.reader.termVectors().get(0)
        is not None,
        "analyzer": "Anserini DefaultEnglishAnalyzer: porter, default Lucene stopwords",
        "field": "contents (title + text)",
        "tie_break": "score descending, external document ID lexicographically ascending",
        "self_match": "exclude document ID equal to query ID",
        "query_repetition": "BagOfWordsQueryGenerator (query term frequency preserved)",
    }


def main() -> None:
    """CLI引数に従ってSciFactを検索し、runと実行メタデータを保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", help="省略するとqrelsにある全クエリを検索")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--show", type=int, default=5, help="1件検索時に表示する上位件数"
    )
    parser.add_argument("--output", type=Path, default=Path("results/scifact/bm25.run"))
    args = parser.parse_args()
    if args.top_k < 1 or args.show < 1:
        parser.error("--top-kと--showは正の整数で指定してください")
    if not math.isfinite(args.k1) or args.k1 < 0 or not 0 <= args.b <= 1:
        parser.error("k1は有限の非負値、bは0以上1以下で指定してください")

    # Set a small JVM before importing Pyserini; never start PyTorch for BM25.
    import jnius_config

    jnius_config.add_options("-Xmx768m", "-XX:ActiveProcessorCount=2")
    from pyserini.prebuilt_index_info import TF_INDEX_INFO
    from pyserini.search import get_qrels, get_topics
    from pyserini.search.lucene import LuceneSearcher

    queries, qrels = normalize_data(get_topics(COLLECTION), get_qrels(COLLECTION))
    if args.query_id is not None and args.query_id not in qrels:
        parser.error(f"評価対象にないクエリIDです: {args.query_id}")
    qids = [args.query_id] if args.query_id is not None else sorted(qrels)
    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    if searcher is None:
        raise RuntimeError(f"索引を読み込めません: {INDEX}")
    analyzer = configure_bm25(searcher, args.k1, args.b)
    metadata = runtime_metadata(searcher, TF_INDEX_INFO[INDEX])
    if not metadata["has_document_vectors"]:
        raise RuntimeError("RM3との共通利用に必要な文書ベクトルがありません")
    metadata.update(
        created_at=datetime.now(UTC).isoformat(),
        collection=COLLECTION,
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
        query_ids=qids,
        # Retain the exact inputs as well as the prebuilt collection identifiers.
        queries={qid: queries[qid] for qid in qids},
        qrels={qid: qrels[qid] for qid in qids},
        implementation_sha256=hashlib.sha256(
            Path(__file__).with_name("bm25_baseline.py").read_bytes()
        ).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: dict[str, int] = {}
    start = time.monotonic()
    with temporary.open("w", encoding="utf-8") as stream:
        for number, qid in enumerate(qids, 1):
            hits = search_query(searcher, queries[qid], qid, args.top_k)
            stream.write(format_run(qid, hits))
            counts[qid] = len(hits)
            if args.query_id is not None:
                preview = {
                    "qid": qid,
                    "query": queries[qid],
                    "analyzed_terms": analyzer.analyze(queries[qid]),
                    "hits": [hit.__dict__ for hit in hits[: args.show]],
                    "score_breakdown": score_breakdown(
                        searcher.index_reader,
                        analyzer,
                        queries[qid],
                        hits[0],
                        args.k1,
                        args.b,
                    )
                    if hits
                    else None,
                }
                print(json.dumps(preview, ensure_ascii=False, indent=2))
            elif number % 50 == 0:
                print(f"searched {number}/{len(qids)}", flush=True)
    temporary.replace(args.output)
    metadata.update(
        elapsed_seconds=time.monotonic() - start,
        hit_counts=counts,
        zero_hit_query_ids=[qid for qid, count in counts.items() if count == 0],
        short_query_ids=[qid for qid, count in counts.items() if count < args.top_k],
        run_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(qids)} queries: {args.output} (metadata: {metadata_path})")


if __name__ == "__main__":
    main()
