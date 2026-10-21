"""Compare plain BM25 with unweighted SPLADE query expansion on BEIR datasets."""

import argparse
import csv
import heapq
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9]+")
WORD = re.compile(r"[a-z]+\Z")


def terms(text):
    return set(TOKEN.findall(text.lower()))


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def load_data(data_dir, dataset):
    base = data_dir / dataset
    documents = {
        row["_id"]: f'{row["title"]} {row["text"]}'
        for row in read_jsonl(base / "corpus.jsonl")
    }
    questions = {row["_id"]: row["text"] for row in read_jsonl(base / "queries.jsonl")}
    gold = defaultdict(dict)
    with (base / "qrels" / "test.tsv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if int(row["score"]) > 0:
                gold[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return documents, questions, gold


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.ids = list(documents)
        self.postings = defaultdict(list)
        lengths = []
        for index, doc_id in enumerate(self.ids):
            counts = Counter(TOKEN.findall(documents[doc_id].lower()))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.avg_length = statistics.mean(lengths)
        self.lengths = lengths

    def search(self, query_terms, limit=10):
        scores = defaultdict(float)
        total = len(self.ids)
        for term in query_terms:
            posting = self.postings.get(term, ())
            if not posting:
                continue
            idf = math.log1p((total - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = heapq.nlargest(limit, scores.items(), key=lambda item: (item[1], -item[0]))
        return [self.ids[index] for index, _ in ranked]


def score(ranked, gold):
    gains = [gold.get(doc_id, 0) for doc_id in ranked[:10]]
    dcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return {"ndcg@10": dcg / idcg if idcg else 0.0,
            "recall@10": sum(gain > 0 for gain in gains) / len(gold)}


def make_expansions(questions, vocabulary, model_dir, max_terms, min_term_length):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir, use_safetensors=False).eval()
    expansions = {}
    with torch.inference_mode():
        for count, (query_id, query) in enumerate(questions.items(), 1):
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
            logits = model(**inputs).logits
            weights = torch.log1p(torch.relu(logits))
            pooled = (weights * inputs["attention_mask"].unsqueeze(-1)).amax(dim=1)[0]
            values = pooled.tolist()
            original = terms(query)
            selected = []
            for index in sorted(range(len(values)), key=lambda i: values[i], reverse=True):
                if values[index] <= 0 or len(selected) >= max_terms:
                    break
                token = tokenizer.convert_ids_to_tokens(index)
                if (WORD.fullmatch(token) and len(token) >= min_term_length
                        and token not in original and token in vocabulary):
                    selected.append({"term": token, "splade_weight": round(values[index], 5)})
            expansions[query_id] = selected
            if count % 50 == 0:
                print(f"encoded {count}/{len(questions)} queries", flush=True)
    return expansions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", choices=("scifact", "fiqa", "arguana", "nfcorpus"), default="scifact")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "model")
    parser.add_argument("--expansions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="First N test queries; 0 means all")
    parser.add_argument("--terms", type=int, default=3, help="Added terms per query")
    parser.add_argument("--min-term-length", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or args.terms < 1 or args.min_term_length < 1:
        parser.error("Require limit >= 0, terms >= 1 and min-term-length >= 1")
    if args.expansions is None:
        if args.dataset == "scifact" and args.terms == 3 and args.min_term_length == 2:
            args.expansions = ROOT / "results" / "expansions.json"
        else:
            args.expansions = ROOT / "results" / args.dataset / (
                f"expansions_{args.terms}terms_min{args.min_term_length}.json"
            )
    if args.output is None:
        if args.dataset == "scifact" and args.terms == 3 and args.min_term_length == 2:
            args.output = ROOT / "results" / "comparison.json"
        else:
            args.output = ROOT / "results" / args.dataset / (
                f"comparison_{args.terms}terms_min{args.min_term_length}.json"
            )

    started = time.perf_counter()
    documents, all_questions, gold = load_data(args.data_dir, args.dataset)
    questions = {qid: all_questions[qid] for qid in gold}
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
    index = BM25(documents)
    if args.expansions.exists():
        expansions = json.loads(args.expansions.read_text(encoding="utf-8"))
    else:
        expansions = {}
    missing = {qid: query for qid, query in questions.items() if qid not in expansions}
    if missing:
        additions = make_expansions(
            missing, index.postings, args.model_dir, args.terms, args.min_term_length
        )
        expansions.update(additions)
        args.expansions.parent.mkdir(parents=True, exist_ok=True)
        args.expansions.write_text(json.dumps(expansions, indent=2), encoding="utf-8")

    rows = []
    for qid, query in questions.items():
        original = terms(query)
        added = [item["term"] for item in expansions[qid][:args.terms]]
        baseline = score(index.search(original), gold[qid])
        expanded = score(index.search(original | set(added)), gold[qid])
        rows.append({"query_id": qid, "query": query, "expansion": added,
                     "baseline": baseline, "expanded": expanded})
    metrics = {
        metric: {"baseline": round(statistics.mean(row["baseline"][metric] for row in rows), 5),
                 "expanded": round(statistics.mean(row["expanded"][metric] for row in rows), 5)}
        for metric in ("ndcg@10", "recall@10")
    }
    for values in metrics.values():
        values["delta"] = round(values["expanded"] - values["baseline"], 5)
    differences = [
        row["expanded"]["ndcg@10"] - row["baseline"]["ndcg@10"] for row in rows
    ]
    result = {"dataset": f"BEIR {args.dataset} test", "documents": len(documents),
              "queries": len(rows), "model": "naver/splade-cocondenser-ensembledistil",
              "added_terms": args.terms, "min_term_length": args.min_term_length,
              "metrics": metrics,
              "ndcg_queries": {"improved": sum(d > 0 for d in differences),
                               "worsened": sum(d < 0 for d in differences),
                               "unchanged": sum(d == 0 for d in differences)},
              "seconds": round(time.perf_counter() - started, 2), "per_query": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
