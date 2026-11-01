"""Run NeuML PubMedBERT SPLADE query terms on the SciFact stemmed BM25 setup."""

import json
import statistics

from experiment import ROOT, load_data
from stemmed_experiment import (ALPHA, REMOVED_WORDS, StemmedBM25, analyzed_terms,
                                measure, regenerate_expansions)

MODEL_REPO = "NeuML/pubmedbert-base-splade"
REVISION = "f284fcbafe4761108f27357f5278d846a630059e"
MODEL_DIR = ROOT / "data" / "model_pubmed"
RESULT_DIR = ROOT / "results" / "scifact"
EXPANSIONS_PATH = RESULT_DIR / "pubmed_filtered_expansions.json"
COMPARISON_PATH = RESULT_DIR / "pubmed_comparison.json"
METRICS = ("ndcg@10", "recall@10", "recall@100")


def main():
    documents, queries, gold = load_data(ROOT / "data", "scifact")
    index = StemmedBM25(documents)
    settings = {"model_repo": MODEL_REPO, "revision": REVISION, "alpha": ALPHA,
                "stemmer": "NLTK Porter original algorithm", "max_terms": 3,
                "removed_words": sorted(REMOVED_WORDS)}
    cache = json.loads(EXPANSIONS_PATH.read_text(encoding="utf-8")) if EXPANSIONS_PATH.exists() else {}
    if cache.get("settings") == settings:
        expansions = cache["expansions"]
    else:
        test_queries = {qid: queries[qid] for qid in gold}
        expansions = regenerate_expansions("scifact-pubmed", test_queries, index, MODEL_DIR)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        EXPANSIONS_PATH.write_text(json.dumps({"settings": settings, "expansions": expansions}, indent=2),
                                   encoding="utf-8")

    reference = json.loads((RESULT_DIR / "stemmed_comparison.json").read_text(encoding="utf-8"))
    reference_rows = {row["query_id"]: row for row in reference["per_query"]}
    rows = []
    for count, (qid, relevant) in enumerate(gold.items(), 1):
        added = expansions[qid]
        pubmed_metrics = measure(index, set(analyzed_terms(queries[qid])),
                                 [item["stem"] for item in added], relevant, None)
        ref = reference_rows[qid]
        baseline = ref["conditions"]["base"]
        marco_metrics = ref["conditions"]["filtered_top3"]
        assert baseline == measure(index, set(analyzed_terms(queries[qid])), [], relevant, None)
        rows.append({"query_id": qid, "query": queries[qid], "pubmed_expansion": added,
                     "marco_expansion": ref["filtered_expansion"],
                     "baseline": baseline, "marco_filtered": marco_metrics,
                     "pubmed_filtered": pubmed_metrics})
        if count % 100 == 0:
            print(f"scifact-pubmed: evaluated {count}/{len(gold)}", flush=True)

    summary = {
        condition: {metric: round(statistics.mean(row[condition][metric] for row in rows), 5)
                    for metric in METRICS}
        for condition in ("baseline", "marco_filtered", "pubmed_filtered")
    }
    deltas = [row["pubmed_filtered"]["ndcg@10"] - row["baseline"]["ndcg@10"]
              for row in rows]
    compared = [row["pubmed_filtered"]["ndcg@10"]
                - row["marco_filtered"]["ndcg@10"] for row in rows]
    def counts(values):
        return {"improved": sum(value > 1e-12 for value in values),
                "unchanged": sum(abs(value) <= 1e-12 for value in values),
                "worsened": sum(value < -1e-12 for value in values)}
    result = {"dataset": "scifact", "queries": len(rows), "model": settings,
              "summary": summary, "pubmed_vs_baseline_queries": counts(deltas),
              "pubmed_vs_marco_queries": counts(compared), "per_query": rows}
    COMPARISON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
