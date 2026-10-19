"""Run the BM25 baseline on LegalBench-RAG ContractNLI."""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from bm25 import BM25, make_chunks
from evaluation import evaluate
from legalbenchrag_io import load_contractnli


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
  parser.add_argument(
    "--chunk-size", type=int, default=200, help="Word tokens per chunk"
  )
  parser.add_argument(
    "--overlap", type=int, default=40, help="Word tokens shared with next chunk"
  )
  parser.add_argument("--top-k", type=int, default=5)
  parser.add_argument(
    "--show-query", type=int, default=0, help="Index of one example query"
  )
  args = parser.parse_args()
  if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size or args.top_k <= 0:
    parser.error("Require chunk-size > overlap >= 0 and top-k > 0")

  documents, questions = load_contractnli(args.data_dir)
  chunks = [
    chunk
    for path, content in documents.items()
    for chunk in make_chunks(path, content, args.chunk_size, args.overlap)
  ]

  started = time.perf_counter()
  index = BM25(chunks)
  index_seconds = time.perf_counter() - started
  rows = {"all_contracts": [], "known_contract": []}
  example = None
  started = time.perf_counter()
  for number, question in enumerate(questions):
    for scope, allowed in (
      ("all_contracts", None),
      ("known_contract", question.gold_document),
    ):
      hits = index.search(question.text, args.top_k, allowed)
      retrieved = [chunks[i] for i, _ in hits]
      rows[scope].append(
        evaluate(question.gold_document, question.gold_ranges, retrieved)
      )
      if number == args.show_query and scope == "all_contracts":
        example = {
          "query": question.text,
          "gold_file": question.gold_document,
          "top_hits": [
            {
              "rank": rank,
              "file": chunks[i].document_id,
              "score": round(score, 3),
              "excerpt": documents[chunks[i].document_id][
                chunks[i].start : chunks[i].end
              ][:240],
            }
            for rank, (i, score) in enumerate(hits, 1)
          ],
        }
  search_seconds = time.perf_counter() - started
  summary = {
    "dataset": "LegalBench-RAG / ContractNLI",
    "questions": len(questions),
    "contracts": len(documents),
    "chunks": len(chunks),
    "parameters": {
      "chunk_size_word_tokens": args.chunk_size,
      "overlap_word_tokens": args.overlap,
      "top_k": args.top_k,
      "bm25_k1": index.k1,
      "bm25_b": index.b,
    },
    "metrics": {
      scope: {
        metric: round(statistics.mean(getattr(row, metric) for row in values), 4)
        for metric in asdict(values[0])
      }
      for scope, values in rows.items()
    },
    "seconds": {
      "index": round(index_seconds, 3),
      "search_and_evaluate": round(search_seconds, 3),
    },
    "example": example,
  }
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
