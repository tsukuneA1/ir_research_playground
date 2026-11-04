"""Evaluate 0, 1, 2, and 3 saved expansion terms with one BM25 index build."""

import argparse
import heapq
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiment import BM25, ROOT, load_data, score, terms


def add_term(index, scores, term):
    posting = index.postings.get(term, ())
    if not posting:
        return
    total = len(index.ids)
    idf = math.log1p((total - len(posting) + 0.5) / (len(posting) + 0.5))
    for doc_index, frequency in posting:
        norm = index.k1 * (
            1 - index.b + index.b * index.lengths[doc_index] / index.avg_length
        )
        scores[doc_index] += idf * frequency * (index.k1 + 1) / (frequency + norm)


def rank(index, scores, exclude_id=None):
    found = heapq.nlargest(
        100,
        ((doc_index, value) for doc_index, value in scores.items()
         if index.ids[doc_index] != exclude_id),
        key=lambda pair: (pair[1], -pair[0]),
    )
    return [index.ids[doc_index] for doc_index, _ in found]


def evaluate(ranked, scores, relevant, id_to_index):
    result = score(ranked, relevant)
    for cutoff in (20, 50, 100):
        result[f"recall@{cutoff}"] = (
            sum(doc_id in relevant for doc_id in ranked[:cutoff]) / len(relevant)
        )
    result["candidate_recall"] = (
        sum(id_to_index[doc_id] in scores for doc_id in relevant) / len(relevant)
    )
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
    per_query = []
    for qid, relevant in gold.items():
        original = terms(questions[qid])
        added = [item["term"] for item in expansions[qid][:3]]
        exclude_id = qid if args.dataset == "arguana" else None
        scores = defaultdict(float)
        for term in sorted(original):
            add_term(index, scores, term)
        original_ranked = rank(index, scores, exclude_id=exclude_id)
        metrics = {"0": evaluate(original_ranked, scores, relevant, id_to_index)}
        top10 = {"0": original_ranked[:10]}
        for count, term in enumerate(added, 1):
            add_term(index, scores, term)
            ranked = rank(index, scores, exclude_id=exclude_id)
            metrics[str(count)] = evaluate(ranked, scores, relevant, id_to_index)
            top10[str(count)] = ranked[:10]
        per_query.append({"query_id": qid, "query": questions[qid],
                          "expansion": added, "metrics": metrics, "top10": top10})
    summary = {
        str(count): {
            metric: round(statistics.mean(row["metrics"][str(count)][metric]
                                          for row in per_query), 5)
            for metric in ("ndcg@10", "recall@10", "recall@20", "recall@50",
                           "recall@100", "candidate_recall")
        }
        for count in range(4)
    }
    result = {"dataset": f"BEIR {args.dataset} test", "documents": len(documents),
              "queries": len(per_query), "prefixes": summary, "per_query": per_query}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"},
                     indent=2))


if __name__ == "__main__":
    main()
