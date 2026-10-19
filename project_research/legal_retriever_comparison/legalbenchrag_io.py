"""LegalBench-RAG JSON adapter. No retrieval or scoring logic lives here."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
  text: str
  gold_document: str
  gold_ranges: list[tuple[int, int]]


def load_contractnli(data_dir: Path) -> tuple[dict[str, str], list[Question]]:
  raw = json.loads(
    (data_dir / "benchmarks/contractnli.json").read_text(encoding="utf-8")
  )
  tests = raw["tests"]
  paths = sorted(
    {snippet["file_path"] for test in tests for snippet in test["snippets"]}
  )
  documents = {
    path: (data_dir / "corpus" / path).read_text(encoding="utf-8") for path in paths
  }
  questions = []
  for test in tests:
    snippets = test["snippets"]
    gold_document = snippets[0]["file_path"]
    gold_ranges = []
    for snippet in snippets:
      path = snippet["file_path"]
      start, end = snippet["span"]
      if path != gold_document or documents[path][start:end] != snippet["answer"]:
        raise ValueError(f"Gold snippet mismatch: {path} [{start}, {end})")
      gold_ranges.append((start, end))
    questions.append(Question(test["query"], gold_document, gold_ranges))
  return documents, questions
