"""Compare similarity based selection among saved top three SPLADE terms."""

import argparse
import difflib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from compare_prefixes import add_term
from experiment import BM25, ROOT, load_data, terms
from weight_and_fallback import evaluate, rank

DATASETS = ("scifact", "fiqa", "arguana", "nfcorpus")
ALPHA = 0.25
STOP = set("a an and are as at be by for from how in is it of on or that the this to was were what when where which who why with".split())
METRICS = ("ndcg@10", "recall@10", "recall@100")


def load_embeddings(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir, use_safetensors=False, local_files_only=True
    ).eval()
    vectors = model.get_input_embeddings().weight.detach().numpy().copy()
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    del model
    return tokenizer, vectors


def similarities(query, candidates, tokenizer, vectors, cache):
    original = sorted(word for word in terms(query) if word.isalpha() and len(word) >= 2 and word not in STOP)
    if not original:
        original = sorted(word for word in terms(query) if word.isalpha())
    original_vectors = []
    for word in original:
        if word not in cache:
            ids = tokenizer.encode(word, add_special_tokens=False)
            vector = vectors[ids].mean(axis=0)
            cache[word] = vector / max(np.linalg.norm(vector), 1e-12)
        original_vectors.append(cache[word])
    matrix = np.stack(original_vectors) if original_vectors else None
    for item in candidates:
        term = item["term"]
        item["lexical_similarity"] = round(
            max((difflib.SequenceMatcher(None, term, word).ratio() for word in original), default=0.0), 5
        )
        token_id = tokenizer.convert_tokens_to_ids(term)
        item["embedding_similarity"] = round(
            float((matrix @ vectors[token_id]).max()) if matrix is not None else 0.0, 5
        )


def choices(candidates):
    if not candidates:
        return {name: [] for name in (
            "all3", "top1", "lexical_high", "lexical_low", "embedding_high", "embedding_low",
            "lexical_weighted", "embedding_weighted"
        )}
    def pick(key, reverse=False):
        return [min(candidates, key=lambda item: (item[key] if reverse else -item[key], item["rank"]))["term"]]
    strongest = candidates[0]["splade_weight"]
    return {
        "all3": [item["term"] for item in candidates],
        "top1": [candidates[0]["term"]],
        "lexical_high": pick("lexical_similarity"),
        "lexical_low": pick("lexical_similarity", reverse=True),
        "embedding_high": pick("embedding_similarity"),
        "embedding_low": pick("embedding_similarity", reverse=True),
        "lexical_weighted": [max(candidates, key=lambda item: (item["splade_weight"] / strongest * item["lexical_similarity"], -item["rank"]))["term"]],
        "embedding_weighted": [max(candidates, key=lambda item: (item["splade_weight"] / strongest * item["embedding_similarity"], -item["rank"]))["term"]],
    }


def run_dataset(dataset, tokenizer, vectors):
    documents, questions, gold = load_data(ROOT / "data", dataset)
    index = BM25(documents)
    expansion_path = (ROOT / "results" / "expansions.json" if dataset == "scifact"
                      else ROOT / "results" / dataset / "expansions_3terms_min2.json")
    expansions = json.loads(expansion_path.read_text(encoding="utf-8"))
    cache = {}
    rows = []
    for count, (qid, relevant) in enumerate(gold.items(), 1):
        query = questions[qid]
        exclude_id = qid if dataset == "arguana" else None
        candidates = [dict(item, rank=rank) for rank, item in enumerate(expansions[qid][:3])]
        similarities(query, candidates, tokenizer, vectors, cache)
        selected = choices(candidates)
        base = defaultdict(float)
        for term in sorted(terms(query)):
            add_term(index, base, term)
        conditions = {"baseline": evaluate(rank(index, base, exclude_id=exclude_id), relevant)}
        extra_scores = {}
        for item in candidates:
            extra = defaultdict(float)
            add_term(index, extra, item["term"])
            extra_scores[item["term"]] = extra
        for name, selected_terms in selected.items():
            scores = base.copy()
            for term in selected_terms:
                for doc_index, value in extra_scores[term].items():
                    scores[doc_index] += ALPHA * value
            conditions[name] = evaluate(rank(index, scores, exclude_id=exclude_id), relevant)
        rows.append({"query_id": qid, "query": query, "candidates": candidates,
                     "selected": selected, "conditions": conditions})
        if count % 200 == 0:
            print(f"{dataset}: evaluated {count}/{len(gold)}", flush=True)
    summary = {
        name: {metric: round(statistics.mean(row["conditions"][name][metric] for row in rows), 5)
               for metric in METRICS}
        for name in rows[0]["conditions"]
    }
    result = {"dataset": dataset, "queries": len(rows), "documents": len(documents),
              "alpha": ALPHA,
              "similarity": "max over non-stopword query tokens; character SequenceMatcher or cosine of SPLADE input embeddings",
              "conditions": summary, "per_query": rows}
    output = ROOT / "results" / dataset / "similarity_filters.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tokenizer, vectors = load_embeddings(ROOT / "data" / "model")
    run_dataset(args.dataset, tokenizer, vectors)


if __name__ == "__main__":
    main()
