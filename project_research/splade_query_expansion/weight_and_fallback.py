"""Test BM25 expansion weights and a genuine zero-result fallback."""

import argparse
import heapq
import json
import statistics
from collections import defaultdict
from pathlib import Path

from compare_prefixes import add_term
from experiment import BM25, ROOT, load_data, score, terms

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def rank(index, scores, cutoff=100, exclude_id=None):
    hits = heapq.nlargest(
        cutoff,
        ((doc_index, value) for doc_index, value in scores.items()
         if value > 0 and index.ids[doc_index] != exclude_id),
        key=lambda pair: (pair[1], -pair[0]),
    )
    return [index.ids[doc_index] for doc_index, _ in hits]


def evaluate(ranked, gold):
    result = score(ranked, gold)
    result["recall@100"] = sum(doc_id in gold for doc_id in ranked[:100]) / len(gold)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scifact", "fiqa", "arguana", "nfcorpus"), required=True)
    parser.add_argument("--expansions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents, questions, gold = load_data(ROOT / "data", args.dataset)
    index = BM25(documents)
    id_to_index = {doc_id: doc_index for doc_index, doc_id in enumerate(index.ids)}
    expansions = json.loads(args.expansions.read_text(encoding="utf-8"))
    rows = []
    for qid, relevant in gold.items():
        exclude_id = qid if args.dataset == "arguana" else None
        excluded_index = id_to_index.get(exclude_id)
        base = defaultdict(float)
        extra = defaultdict(float)
        for term in sorted(terms(questions[qid])):
            add_term(index, base, term)
        added = [item["term"] for item in expansions[qid][:3]]
        for term in added:
            add_term(index, extra, term)
        conditions = {}
        for weight in WEIGHTS:
            scores = base.copy()
            if weight:
                for doc_index, value in extra.items():
                    scores[doc_index] += weight * value
            conditions[str(weight)] = evaluate(rank(index, scores, exclude_id=exclude_id), relevant)
        # This trigger is observable without relevance labels.
        baseline_candidate_count = len(base) - int(excluded_index in base)
        zero_hit = baseline_candidate_count == 0
        fallback = conditions["1.0"] if zero_hit else conditions["0.0"]
        rows.append({"query_id": qid, "query": questions[qid],
                     "expansion": added, "zero_hit": zero_hit,
                     "baseline_candidate_count": baseline_candidate_count,
                     "conditions": conditions, "fallback": fallback})

    metrics = ("ndcg@10", "recall@10", "recall@100")
    summary = {
        weight: {
            metric: round(statistics.mean(row["conditions"][weight][metric]
                                          for row in rows), 5)
            for metric in metrics
        }
        for weight in map(str, WEIGHTS)
    }
    fallback_summary = {
        metric: round(statistics.mean(row["fallback"][metric] for row in rows), 5)
        for metric in metrics
    }
    oracle_zero = [row for row in rows
                   if row["conditions"]["0.0"]["recall@10"] == 0]
    result = {
        "dataset": f"BEIR {args.dataset} test",
        "documents": len(documents), "queries": len(rows),
        "score_rule": "BM25(original) + weight * BM25(three SPLADE terms)",
        "weights": summary,
        "zero_hit_queries": sum(row["zero_hit"] for row in rows),
        "zero_hit_fallback": fallback_summary,
        "oracle_no_relevant_at_10_queries": len(oracle_zero),
        "oracle_rescued_by_weight_1_at_10": sum(
            row["conditions"]["1.0"]["recall@10"] > 0 for row in oracle_zero
        ),
        "per_query": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"},
                     indent=2))


if __name__ == "__main__":
    main()
