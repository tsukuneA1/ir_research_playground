"""保存したTREC runと実行メタデータを読み、評価結果をJSONに保存する。"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from evaluation import evaluate_run


def load_run(path: Path) -> dict[str, list[str]]:
    """TREC runの明示的な順位を読み、クエリ別の文書ID列へ変換する。

    空行は無視する。順位は1始まりで連続し、文書IDはクエリ内で重複せず、
    スコアは有限かつ降順であることを確認する。不正な場合はValueError。
    同点でもスコアで並べ直さず、保存されたrankを尊重する。
    """
    rows: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6 or fields[1] != "Q0":
                raise ValueError(f"{path}:{number}: TREC runの6列形式ではありません")
            qid, _, docid, rank, score, _ = fields
            rows[qid].append((int(rank), docid, float(score)))
    rankings = {}
    for qid, hits in rows.items():
        hits.sort(key=lambda hit: hit[0])
        if [hit[0] for hit in hits] != list(range(1, len(hits) + 1)):
            raise ValueError(f"順位が1始まりの連続値ではありません: {qid}")
        if len({hit[1] for hit in hits}) != len(hits):
            raise ValueError(f"文書IDが重複しています: {qid}")
        scores = [hit[2] for hit in hits]
        if not all(map(math.isfinite, scores)) or scores != sorted(
            scores, reverse=True
        ):
            raise ValueError(f"スコアが有限の降順値ではありません: {qid}")
        rankings[qid] = [hit[1] for hit in hits]
    return rankings


def main() -> None:
    """runと隣接するrun.jsonを入力に評価し、平均とクエリ別の値を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/scifact/bm25.run"))
    parser.add_argument(
        "--metadata", type=Path, help="省略時はrunのパスに.jsonを付ける"
    )
    parser.add_argument(
        "--output", type=Path, help="省略時はrunのパスに.metrics.jsonを付ける"
    )
    args = parser.parse_args()
    metadata_path = args.metadata or Path(str(args.run) + ".json")
    output = args.output or Path(str(args.run) + ".metrics.json")
    if output.resolve() in (args.run.resolve(), metadata_path.resolve()):
        parser.error("出力先を入力runまたはメタデータと同じにできません")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_hash = hashlib.sha256(args.run.read_bytes()).hexdigest()
    if run_hash != metadata["run_sha256"]:
        parser.error("runのSHA-256がメタデータと一致しません")
    qrels = {
        str(qid): {str(docid): int(grade) for docid, grade in docs.items()}
        for qid, docs in metadata["qrels"].items()
    }
    query_ids = [str(qid) for qid in metadata["query_ids"]]
    rankings = load_run(args.run)
    if extra := rankings.keys() - set(query_ids):
        parser.error(f"評価対象外のクエリがrunに含まれています: {sorted(extra)}")
    result = evaluate_run(rankings, qrels, query_ids)
    report = {
        "run": str(args.run),
        "run_sha256": run_hash,
        "metadata": str(metadata_path),
        "retrieval_depth": metadata["top_k"],
        "evaluated_queries": len(result.per_query),
        "ndcg_gain": "linear relevance grade; nonpositive and unjudged = 0",
        "relevant_threshold": "grade > 0",
        **asdict(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.mean, ensure_ascii=False, indent=2))
    print(f"Saved {len(result.per_query)} queries: {output}")


if __name__ == "__main__":
    main()
