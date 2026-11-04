"""Compare SPLADE terms against BM25 with English stemming and stop words."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache

from nltk.stem import PorterStemmer

from compare_prefixes import add_term
from experiment import ROOT, load_data, score
from weight_and_fallback import rank

DATASETS = ("scifact", "fiqa", "arguana", "nfcorpus")
TOKEN = re.compile(r"[a-z0-9]+")
WORD = re.compile(r"[a-z]+\Z")
ALPHA = 0.25
STOP_WORDS = frozenset("""
    a an and are as at be but by for if in into is it of on or such
    that the their then there these they this to was will with
""".split())
DISCOURSE_WORDS = frozenset("""
    because although however therefore moreover nevertheless whereas while
    unless thus also so yet though since furthermore consequently
""".split())
REMOVED_WORDS = STOP_WORDS | DISCOURSE_WORDS
STEMMER = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


@lru_cache(maxsize=200_000)
def stem(word):
    return STEMMER.stem(word)


def analyzed_terms(text):
    return [stem(word) for word in TOKEN.findall(text.lower())
            if word not in REMOVED_WORDS]


class StemmedBM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.ids = list(documents)
        self.postings = defaultdict(list)
        self.lengths = []
        for doc_index, doc_id in enumerate(self.ids):
            counts = Counter(analyzed_terms(documents[doc_id]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((doc_index, frequency))
        self.avg_length = statistics.mean(self.lengths)


def expansion_file(dataset, filtered):
    name = "stemmed_filtered_expansions.json" if filtered else "expansions_3terms_min2.json"
    if not filtered and dataset == "scifact":
        return ROOT / "results" / "expansions.json"
    return ROOT / "results" / dataset / name


def regenerate_expansions(dataset, questions, index, model_dir=None):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_dir = model_dir or ROOT / "data" / "model"
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).eval()
    expansions = {}
    with torch.inference_mode():
        for count, (qid, query) in enumerate(questions.items(), 1):
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
            logits = model(**inputs).logits
            pooled = (torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)).amax(dim=1)[0]
            original_stems = set(analyzed_terms(query))
            selected = []
            values, indices = torch.topk(pooled, min(1024, pooled.numel()))
            for value, token_id in zip(values.tolist(), indices.tolist()):
                if value <= 0 or len(selected) == 3:
                    break
                token = tokenizer.convert_ids_to_tokens(token_id)
                if not (WORD.fullmatch(token) and len(token) >= 2):
                    continue
                if token in REMOVED_WORDS:
                    continue
                normalized = stem(token)
                if normalized in original_stems or normalized not in index.postings:
                    continue
                if any(item["stem"] == normalized for item in selected):
                    continue
                selected.append({"term": token, "stem": normalized,
                                 "splade_weight": round(value, 5)})
            expansions[qid] = selected
            if count % 100 == 0:
                print(f"{dataset}: regenerated {count}/{len(questions)}", flush=True)
    return expansions


def measure(index, query_stems, added_stems, relevant, exclude_id, alpha=ALPHA):
    scores = defaultdict(float)
    for term in sorted(query_stems):
        add_term(index, scores, term)
    if added_stems:
        for term in added_stems:
            extra = defaultdict(float)
            add_term(index, extra, term)
            for doc_index, value in extra.items():
                scores[doc_index] += alpha * value
    ranked = rank(index, scores, cutoff=100, exclude_id=exclude_id)
    result = score(ranked, relevant)
    result["recall@100"] = sum(doc_id in relevant for doc_id in ranked) / len(relevant)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()
    if args.alpha < 0:
        parser.error("alpha must be nonnegative")
    documents, questions, gold = load_data(ROOT / "data", args.dataset)
    index = StemmedBM25(documents)
    filtered_path = expansion_file(args.dataset, True)
    settings = {"stemmer": "NLTK Porter original algorithm", "removed_words": sorted(REMOVED_WORDS),
                "max_terms": 3, "min_length": 2}
    cache = json.loads(filtered_path.read_text(encoding="utf-8")) if filtered_path.exists() else {}
    if cache.get("settings") == settings:
        filtered = cache["expansions"]
    else:
        test_questions = {qid: questions[qid] for qid in gold}
        filtered = regenerate_expansions(args.dataset, test_questions, index)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_path.write_text(json.dumps({"settings": settings, "expansions": filtered}, indent=2),
                                 encoding="utf-8")
    old = json.loads(expansion_file(args.dataset, False).read_text(encoding="utf-8"))

    rows = []
    for count, (qid, relevant) in enumerate(gold.items(), 1):
        original = set(analyzed_terms(questions[qid]))
        original_terms = [item["term"] for item in old[qid][:3]]
        filtered_terms = [item["term"] for item in filtered[qid]]
        conditions = {
            "base": measure(index, original, [], relevant,
                            qid if args.dataset == "arguana" else None, args.alpha),
            "old_top3": measure(index, original,
                                [stem(term) for term in original_terms if term not in REMOVED_WORDS],
                                relevant, qid if args.dataset == "arguana" else None, args.alpha),
            "filtered_top3": measure(index, original,
                                     [item["stem"] for item in filtered[qid]], relevant,
                                     qid if args.dataset == "arguana" else None, args.alpha),
        }
        rows.append({"query_id": qid, "query": questions[qid],
                     "old_expansion": original_terms, "filtered_expansion": filtered_terms,
                     "conditions": conditions})
        if count % 200 == 0:
            print(f"{args.dataset}: evaluated {count}/{len(gold)}", flush=True)

    metrics = ("ndcg@10", "recall@10", "recall@100")
    summary = {
        condition: {metric: round(statistics.mean(row["conditions"][condition][metric]
                                                 for row in rows), 5)
                    for metric in metrics}
        for condition in ("base", "old_top3", "filtered_top3")
    }
    delta = [row["conditions"]["filtered_top3"]["ndcg@10"]
             - row["conditions"]["base"]["ndcg@10"] for row in rows]
    result = {"dataset": args.dataset, "queries": len(rows), "documents": len(documents),
              "alpha": args.alpha, "stemmer": "NLTK Porter original algorithm",
              "removed_words": sorted(REMOVED_WORDS), "summary": summary,
              "filtered_expansion_count": Counter(len(filtered[qid]) for qid in gold),
              "filtered_vs_base_queries": {
                  "improved": sum(value > 1e-12 for value in delta),
                  "unchanged": sum(abs(value) <= 1e-12 for value in delta),
                  "worsened": sum(value < -1e-12 for value in delta),
              }, "per_query": rows}
    output_name = ("stemmed_comparison.json" if args.alpha == ALPHA else
                   f"stemmed_comparison_alpha{args.alpha:g}.json")
    output = ROOT / "results" / args.dataset / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
