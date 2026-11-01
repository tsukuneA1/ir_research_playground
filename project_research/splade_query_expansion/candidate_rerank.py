"""Separate candidate-pool and ranking effects of SPLADE terms added to BM25."""

import argparse
import heapq
import json
import statistics
from collections import defaultdict
from pathlib import Path

from compare_prefixes import add_term
from experiment import BM25, ROOT, load_data, score, terms

DATASETS = ("scifact", "fiqa", "arguana", "nfcorpus")
CUTOFFS = (100, 1000)
CONDITIONS = ("base_pool_base_rank", "base_pool_expanded_rank",
              "expanded_pool_base_rank", "expanded_pool_expanded_rank")
METRICS = ("ndcg@10", "recall@10", "recall@100")


def top_pool(index, scores, cutoff, excluded_id):
    return [doc_index for doc_index, _ in heapq.nlargest(
        cutoff,
        ((doc_index, value) for doc_index, value in scores.items()
         if value > 0 and index.ids[doc_index] != excluded_id),
        key=lambda item: (item[1], -item[0]),
    )]


def evaluate_pool(index, pool, scores, relevant):
    top = heapq.nlargest(
        100, pool, key=lambda doc_index: (scores.get(doc_index, 0.0), -doc_index)
    )
    ranked = [index.ids[doc_index] for doc_index in top]
    metrics = score(ranked, relevant)
    metrics["recall@100"] = sum(doc_id in relevant for doc_id in ranked) / len(relevant)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    if args.alpha < 0:
        parser.error("alpha must be nonnegative")

    documents, questions, gold = load_data(ROOT / "data", args.dataset)
    index = BM25(documents)
    expansion_path = (ROOT / "results" / "expansions.json" if args.dataset == "scifact"
                      else ROOT / "results" / args.dataset / "expansions_3terms_min2.json")
    expansions = json.loads(expansion_path.read_text(encoding="utf-8"))
    id_to_index = {doc_id: doc_index for doc_index, doc_id in enumerate(index.ids)}
    rows = []
    for count, (qid, relevant) in enumerate(gold.items(), 1):
        excluded_id = qid if args.dataset == "arguana" else None
        base = defaultdict(float)
        extra = defaultdict(float)
        for term in sorted(terms(questions[qid])):
            add_term(index, base, term)
        added = [item["term"] for item in expansions[qid][:3]]
        for term in added:
            add_term(index, extra, term)
        expanded = base.copy()
        for doc_index, value in extra.items():
            expanded[doc_index] += args.alpha * value

        base_top = top_pool(index, base, max(CUTOFFS), excluded_id)
        expanded_top = top_pool(index, expanded, max(CUTOFFS), excluded_id)
        relevant_indices = {id_to_index[doc_id] for doc_id in relevant if doc_id in id_to_index}
        cutoffs = {}
        for cutoff in CUTOFFS:
            base_pool = base_top[:cutoff]
            expanded_pool = expanded_top[:cutoff]
            base_set = set(base_pool)
            expanded_set = set(expanded_pool)
            new = expanded_set - base_set
            lost = base_set - expanded_set
            conditions = {
                "base_pool_base_rank": evaluate_pool(index, base_pool, base, relevant),
                "base_pool_expanded_rank": evaluate_pool(index, base_pool, expanded, relevant),
                "expanded_pool_base_rank": evaluate_pool(index, expanded_pool, base, relevant),
                "expanded_pool_expanded_rank": evaluate_pool(index, expanded_pool, expanded, relevant),
            }
            cutoffs[str(cutoff)] = {
                "candidate_recall": {
                    "base": len(base_set & relevant_indices) / len(relevant),
                    "expanded": len(expanded_set & relevant_indices) / len(relevant),
                },
                "new_documents": len(new),
                "lost_documents": len(lost),
                "new_relevant": len(new & relevant_indices),
                "lost_relevant": len(lost & relevant_indices),
                "conditions": conditions,
            }
        rows.append({"query_id": qid, "query": questions[qid], "expansion": added,
                     "cutoffs": cutoffs})
        if count % 200 == 0:
            print(f"{args.dataset}: evaluated {count}/{len(gold)}", flush=True)

    summary = {}
    for cutoff in CUTOFFS:
        key = str(cutoff)
        summary[key] = {
            "candidate_recall": {
                condition: round(statistics.mean(row["cutoffs"][key]["candidate_recall"][condition]
                                                  for row in rows), 5)
                for condition in ("base", "expanded")
            },
            "queries_with_new_relevant": sum(row["cutoffs"][key]["new_relevant"] > 0 for row in rows),
            "queries_with_lost_relevant": sum(row["cutoffs"][key]["lost_relevant"] > 0 for row in rows),
            "total_new_relevant": sum(row["cutoffs"][key]["new_relevant"] for row in rows),
            "total_lost_relevant": sum(row["cutoffs"][key]["lost_relevant"] for row in rows),
            "conditions": {
                condition: {
                    metric: round(statistics.mean(row["cutoffs"][key]["conditions"][condition][metric]
                                                  for row in rows), 5)
                    for metric in METRICS
                }
                for condition in CONDITIONS
            },
        }
    result = {"dataset": args.dataset, "queries": len(rows), "documents": len(documents),
              "alpha": args.alpha, "added_terms": 3, "cutoffs": summary, "per_query": rows}
    output = ROOT / "results" / args.dataset / "candidate_rerank.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
