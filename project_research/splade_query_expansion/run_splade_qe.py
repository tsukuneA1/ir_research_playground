"""SciFactのSPLADE QEを順次実行し、推論キャッシュ・追加語・runを保存する。"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from bm25_baseline import configure_bm25
from run_bm25 import COLLECTION, INDEX, format_run, runtime_metadata
from splade_model import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION, load_splade
from splade_qe import search_with_splade
from splade_query import SparseQuery, TokenWeight


def cache_identity() -> dict[str, str | int]:
    """推論結果を再利用できるモデル・入力長・実装・依存ライブラリを識別する。"""
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "pooling": "max(log1p(relu(logits)) * attention_mask)",
        "dtype": "float32",
        "device": "cpu",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "encoder_sha256": hashlib.sha256(
            Path(__file__).with_name("splade_query.py").read_bytes()
        ).hexdigest(),
    }


def load_cached_query(
    path: Path, identity: dict[str, str | int], query: str
) -> SparseQuery | None:
    """同じ推論条件・原文のキャッシュだけを復元する。未取得や条件違いはNone。"""
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["identity"] != identity or record["representation"]["query"] != query:
        return None
    fields = record["representation"].copy()
    fields["terms"] = [TokenWeight(**term) for term in fields["terms"]]
    return SparseQuery(**fields)


def save_cached_query(
    path: Path, identity: dict[str, str | int], encoded: SparseQuery
) -> None:
    """全正重みをk・λに依存しない形で保存し、中断後もクエリ単位で再利用する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"identity": identity, "representation": asdict(encoded)},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    """指定k・λでSciFactを再検索し、既存の評価CLIで読めるrunとJSONを保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True, help="選別後の追加索引語数")
    parser.add_argument("--lambda", dest="expansion_weight", type=float, required=True)
    parser.add_argument("--weighting", choices=("splade", "uniform"), default="splade")
    parser.add_argument("--top-k", type=int, default=1000, help="取得文書数")
    parser.add_argument(
        "--query-id", action="append", help="省略すると全評価対象クエリ"
    )
    parser.add_argument(
        "--baseline-metadata", type=Path, default=Path("results/scifact/bm25.run.json")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("results/scifact/splade-cache")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k < 1 or args.top_k < 1:
        parser.error("kとtop-kは正の整数で指定してください")
    if not math.isfinite(args.expansion_weight) or args.expansion_weight < 0:
        parser.error("λは有限の非負値で指定してください")
    baseline = json.loads(args.baseline_metadata.read_text(encoding="utf-8"))
    if baseline["collection"] != COLLECTION or baseline["index"] != INDEX:
        parser.error("SciFact testのBM25メタデータを指定してください")
    qids = args.query_id or baseline["query_ids"]
    if len(qids) != len(set(qids)) or set(qids) - set(baseline["query_ids"]):
        parser.error("クエリIDが重複しているか、ベースラインの評価対象にありません")
    tag = f"splade-{args.weighting}-k{args.k}-lambda{args.expansion_weight:g}"
    output = args.output or Path(f"results/scifact/{tag}.run")
    if output.resolve() in (
        args.baseline_metadata.resolve(),
        args.baseline_metadata.with_suffix("").resolve(),
    ):
        parser.error("出力先をベースラインのrun・メタデータと同じにできません")

    # JVMを制限して起動し、モデルは未推論のクエリがある場合だけ読み込む。
    import jnius_config

    jnius_config.add_options("-Xmx768m", "-XX:ActiveProcessorCount=2")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(baseline["index_path"])
    analyzer = configure_bm25(searcher, baseline["k1"], baseline["b"])
    metadata = runtime_metadata(
        searcher,
        {"filename": baseline["index_archive"], "md5": baseline["index_archive_md5"]},
    )
    identity = cache_identity()
    metadata.update(
        created_at=datetime.now(UTC).isoformat(),
        collection=COLLECTION,
        baseline_metadata=str(args.baseline_metadata),
        k1=baseline["k1"],
        b=baseline["b"],
        k=args.k,
        expansion_weight=args.expansion_weight,
        weighting=args.weighting,
        top_k=args.top_k,
        query_ids=qids,
        queries={qid: baseline["queries"][qid] for qid in qids},
        qrels={qid: baseline["qrels"][qid] for qid in qids},
        splade=identity,
        cache_dir=str(args.cache_dir),
        implementation_hashes={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "splade_qe.py",
                "splade_model.py",
                "bm25_baseline.py",
                "run_splade_qe.py",
            )
        },
    )
    loaded = None
    cache_hits = 0
    per_query = {}
    counts = {}
    start = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for number, qid in enumerate(qids, 1):
            query = baseline["queries"][qid]
            cache_path = args.cache_dir / f"{qid}.json"
            encoded = load_cached_query(cache_path, identity, query)
            cached = encoded is not None
            if not cached and loaded is None:
                print("取得済みSPLADE-v3をCPUに読み込みます。", flush=True)
                loaded = load_splade(local_files_only=True)
            query_start = time.perf_counter()
            result = search_with_splade(
                loaded,
                searcher,
                analyzer,
                query,
                qid,
                k=args.k,
                expansion_weight=args.expansion_weight,
                weighting=args.weighting,
                top_k=args.top_k,
                encoded=encoded,
            )
            if not cached:
                save_cached_query(cache_path, identity, result.sparse_query)
            cache_hits += int(cached)
            counts[qid] = len(result.hits)
            per_query[qid] = {
                "expansion_terms": [asdict(term) for term in result.expansion_terms],
                "lucene_query": result.lucene_query,
                "original_token_count": result.sparse_query.original_token_count,
                "used_token_count": result.sparse_query.used_token_count,
                "truncated": result.sparse_query.truncated,
                "from_cache": cached,
                "elapsed_seconds": time.perf_counter() - query_start,
            }
            stream.write(format_run(qid, result.hits, tag))
            if number % 25 == 0 or number == len(qids):
                print(
                    f"検索済み {number}/{len(qids)}（推論キャッシュ利用 {cache_hits}）",
                    flush=True,
                )
    temporary.replace(output)
    metadata.update(
        elapsed_seconds=time.perf_counter() - start,
        inference_cache_hits=cache_hits,
        per_query=per_query,
        hit_counts=counts,
        zero_hit_query_ids=[qid for qid, count in counts.items() if count == 0],
        short_query_ids=[qid for qid, count in counts.items() if count < args.top_k],
        no_expansion_query_ids=[
            qid for qid, row in per_query.items() if not row["expansion_terms"]
        ],
        run_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    metadata_path = Path(str(output) + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {output} (metadata: {metadata_path})")


if __name__ == "__main__":
    main()
