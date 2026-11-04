"""Export a query-level index and inspectable examples for the candidate/rerank run."""

import csv
import heapq
import json
from collections import defaultdict

from candidate_rerank import DATASETS, top_pool
from compare_prefixes import add_term
from experiment import BM25, ROOT, load_data, score, terms

OUTPUT = ROOT / "analysis_data"
ALPHA = 0.25
SELECTED = {
    "scifact": {
        "212": "large_improvement",
        "660": "large_decline",
        "1196": "unchanged_with_new_relevant_candidate",
    },
    "fiqa": {
        "2388": "large_improvement",
        "691": "large_decline",
        "10122": "unchanged_with_new_relevant_candidate",
    },
    "arguana": {
        "test-sport-tshbmlbscac-con02a": "large_improvement",
        "test-culture-mmctghwbsa-pro03a": "large_decline",
        "test-health-dhghhbampt-con01a": "unchanged_with_new_relevant_candidate",
    },
    "nfcorpus": {
        "PLAIN-2386": "improvement_with_new_relevant_candidates",
        "PLAIN-2600": "improvement_without_new_relevant_candidates",
        "PLAIN-3085": "decline_without_lost_relevant_candidates",
        "PLAIN-3171": "decline_with_lost_relevant_candidate",
        "PLAIN-112": "unchanged_with_new_relevant_candidate",
    },
}


def change_label(delta):
    if delta > 1e-12:
        return "improved"
    if delta < -1e-12:
        return "worsened"
    return "unchanged"


def query_index_rows():
    rows = []
    selected = {}
    for dataset in DATASETS:
        result = json.loads((ROOT / "results" / dataset / "candidate_rerank.json")
                            .read_text(encoding="utf-8"))
        if result["alpha"] != ALPHA or result["added_terms"] != 3:
            raise ValueError(f"Unexpected experiment settings for {dataset}")
        for item in result["per_query"]:
            c100 = item["cutoffs"]["100"]
            c1000 = item["cutoffs"]["1000"]
            conditions = c100["conditions"]
            base = conditions["base_pool_base_rank"]
            ranked = conditions["base_pool_expanded_rank"]
            candidate = conditions["expanded_pool_base_rank"]
            full = conditions["expanded_pool_expanded_rank"]
            delta = full["ndcg@10"] - base["ndcg@10"]
            row = {
                "dataset": dataset,
                "query_id": item["query_id"],
                "change": change_label(delta),
                "query": item["query"],
                "expansion_terms": " | ".join(item["expansion"]),
                "base_ndcg10": round(base["ndcg@10"], 6),
                "fixed_candidates_expanded_rank_ndcg10": round(ranked["ndcg@10"], 6),
                "expanded_candidates_base_rank_ndcg10": round(candidate["ndcg@10"], 6),
                "expanded_ndcg10": round(full["ndcg@10"], 6),
                "delta_ndcg10": round(delta, 6),
                "base_recall10": round(base["recall@10"], 6),
                "expanded_recall10": round(full["recall@10"], 6),
                "base_candidate_recall100": round(c100["candidate_recall"]["base"], 6),
                "expanded_candidate_recall100": round(c100["candidate_recall"]["expanded"], 6),
                "new_relevant_in_candidates100": c100["new_relevant"],
                "lost_relevant_from_candidates100": c100["lost_relevant"],
                "base_candidate_recall1000": round(c1000["candidate_recall"]["base"], 6),
                "expanded_candidate_recall1000": round(c1000["candidate_recall"]["expanded"], 6),
            }
            rows.append(row)
            if item["query_id"] in SELECTED[dataset]:
                selected[(dataset, item["query_id"])] = (item, row)
    return rows, selected


def top10_within(pool, scores):
    return heapq.nlargest(10, pool,
                          key=lambda doc_index: (scores.get(doc_index, 0.0), -doc_index))


def round_score(value):
    return round(value, 5)


def inspect_query(index, documents, gold, item, summary_row, selection_reason):
    qid = item["query_id"]
    relevant = gold[qid]
    excluded_id = qid if summary_row["dataset"] == "arguana" else None
    base = defaultdict(float)
    term_scores = {}
    for term in sorted(terms(item["query"])):
        add_term(index, base, term)
    extra = defaultdict(float)
    for term in item["expansion"]:
        scores = defaultdict(float)
        add_term(index, scores, term)
        term_scores[term] = scores
        for doc_index, value in scores.items():
            extra[doc_index] += value
    expanded = base.copy()
    for doc_index, value in extra.items():
        expanded[doc_index] += ALPHA * value

    base_pool = top_pool(index, base, 1000, excluded_id)
    expanded_pool = top_pool(index, expanded, 1000, excluded_id)
    base_rank = {doc_index: pos for pos, doc_index in enumerate(base_pool, 1)}
    expanded_rank = {doc_index: pos for pos, doc_index in enumerate(expanded_pool, 1)}
    pools = {"base": base_pool[:100], "expanded": expanded_pool[:100]}
    top10 = {
        "base_candidates_base_rank": top10_within(pools["base"], base),
        "base_candidates_expanded_rank": top10_within(pools["base"], expanded),
        "expanded_candidates_base_rank": top10_within(pools["expanded"], base),
        "expanded_candidates_expanded_rank": top10_within(pools["expanded"], expanded),
    }
    focus = set().union(*(set(docs) for docs in top10.values()))
    focus.update(doc_index for doc_index in base_pool[:20] + expanded_pool[:20]
                 if index.ids[doc_index] in relevant)
    relevant_new = [doc_index for doc_index in pools["expanded"]
                    if doc_index not in pools["base"] and index.ids[doc_index] in relevant]
    relevant_lost = [doc_index for doc_index in pools["base"]
                     if doc_index not in pools["expanded"] and index.ids[doc_index] in relevant]
    focus.update(relevant_new + relevant_lost)
    document_details = {}
    for doc_index in sorted(focus):
        doc_id = index.ids[doc_index]
        document_details[doc_id] = {
            "title_or_text_start": documents[doc_id][:220],
            "relevance_grade": relevant.get(doc_id, 0),
            "base_rank_top1000": base_rank.get(doc_index),
            "expanded_rank_top1000": expanded_rank.get(doc_index),
            "base_bm25": round_score(base.get(doc_index, 0.0)),
            "added_terms_bm25": {term: round_score(scores.get(doc_index, 0.0))
                                 for term, scores in term_scores.items()},
            "expanded_bm25": round_score(expanded.get(doc_index, 0.0)),
        }
    result = {
        "dataset": summary_row["dataset"],
        "query_id": qid,
        "selection_reason": selection_reason,
        "summary": summary_row,
        "relevant_document_count": len(relevant),
        "missing_relevant_document_ids": [doc_id for doc_id in relevant if doc_id not in documents],
        "top10": {condition: [index.ids[doc_index] for doc_index in ranked]
                  for condition, ranked in top10.items()},
        "new_relevant_in_candidates100": [index.ids[doc_index] for doc_index in relevant_new],
        "lost_relevant_from_candidates100": [index.ids[doc_index] for doc_index in relevant_lost],
        "documents": document_details,
    }
    expected = item["cutoffs"]["100"]
    assert len(relevant_new) == expected["new_relevant"]
    assert len(relevant_lost) == expected["lost_relevant"]
    for name, ranked in top10.items():
        expected_name = name.replace("candidates", "pool")
        measured = score([index.ids[doc_index] for doc_index in ranked], relevant)
        for metric in ("ndcg@10", "recall@10"):
            assert abs(measured[metric] - expected["conditions"][expected_name][metric]) < 1e-10
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, selected = query_index_rows()
    with (OUTPUT / "query_effects.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases = []
    for dataset in DATASETS:
        documents, questions, gold = load_data(ROOT / "data", dataset)
        index = BM25(documents)
        for qid, reason in SELECTED[dataset].items():
            item, row = selected[(dataset, qid)]
            assert questions[qid] == item["query"]
            case = inspect_query(index, documents, gold, item, row, reason)
            cases.append(case)
            print(f"{dataset} {qid}: {row['change']} {row['delta_ndcg10']:+.5f}", flush=True)
    (OUTPUT / "selected_query_cases.json").write_text(
        json.dumps({"alpha": ALPHA, "candidate_cutoff": 100,
                    "rank_lookahead": 1000, "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"exported {len(rows)} queries and {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()
