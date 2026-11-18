"""数件のクエリをCPU推論し、生のSPLADEトークンと重みを表示する。"""

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from splade_model import load_splade
from splade_query import encode_query

EXAMPLES = [
    "Ivermectin is used to treat onchocerciasis.",
    "How to categorize credit card payments?",
    "What causes lung cancer?",
]


def main() -> None:
    """自由文・SciFactの指定ID・例文を順次推論し、重みと実行時間を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--query", action="append", help="複数回指定可能。省略時は例文3件"
    )
    source.add_argument(
        "--scifact-query-id",
        action="append",
        help="保存済みSciFactのクエリID。複数回指定可能",
    )
    parser.add_argument(
        "--scifact-metadata",
        type=Path,
        default=Path("results/scifact/bm25.run.json"),
        help="クエリ原文を読み込むBM25実行メタデータ",
    )
    parser.add_argument(
        "--show", type=int, default=20, help="表示件数。拡張語のcutoffではない"
    )
    parser.add_argument(
        "--output", type=Path, help="指定した場合、表示内容をJSONにも保存する"
    )
    args = parser.parse_args()
    if args.show < 1:
        parser.error("--showは正の整数で指定してください")
    queries = args.query or EXAMPLES
    query_ids = [None] * len(queries)
    if args.scifact_query_id:
        metadata = json.loads(args.scifact_metadata.read_text(encoding="utf-8"))
        if metadata["collection"] != "beir-v1.0.0-scifact-test":
            parser.error("指定したメタデータはSciFact testのものではありません")
        missing = set(args.scifact_query_id) - metadata["queries"].keys()
        if missing:
            parser.error(f"メタデータにないクエリIDです: {sorted(missing)}")
        query_ids = args.scifact_query_id
        queries = [metadata["queries"][qid] for qid in query_ids]
    if any(not query.strip() for query in queries):
        parser.error("空のクエリは指定できません")
    print("取得済みモデルをCPUに読み込みます。", flush=True)
    start = time.perf_counter()
    loaded = load_splade(local_files_only=True)
    load_seconds = time.perf_counter() - start
    results = []
    for qid, query in zip(query_ids, queries, strict=True):
        start = time.perf_counter()
        encoded = encode_query(loaded, query)
        seconds = time.perf_counter() - start
        result = {
            "query_id": qid,
            "query": query,
            "input_tokens": encoded.input_tokens,
            "original_token_count": encoded.original_token_count,
            "used_token_count": encoded.used_token_count,
            "truncated": encoded.truncated,
            "positive_term_count": len(encoded.terms),
            "encoding_seconds": seconds,
            "top_terms": [asdict(term) for term in encoded.terms[: args.show]],
            "top_non_input_terms": [
                asdict(term)
                for term in [term for term in encoded.terms if not term.in_input][
                    : args.show
                ]
            ],
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    report = {
        "model_id": loaded.model_id,
        "revision": loaded.revision,
        "max_input_tokens": loaded.max_input_tokens,
        "device": str(loaded.model.device),
        "dtype": str(loaded.model.dtype),
        "load_seconds": load_seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "pooling": "max(log1p(relu(logits)) * attention_mask)",
        "show": args.show,
        "query_source": str(args.scifact_metadata)
        if args.scifact_query_id
        else "自由文・例文",
        "results": results,
    }
    print(
        f"モデル読み込み: {load_seconds:.3f}秒、最大RSS: {report['peak_rss_mib']:.1f} MiB"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
