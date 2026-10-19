"""Dataset-independent evaluation against evidence character ranges."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol


class TextSpan(Protocol):
  document_id: str
  start: int
  end: int


@dataclass(frozen=True)
class RetrievalMetrics:
  document_hit: bool
  evidence_hit: bool
  precision: float
  recall: float


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
  merged = []
  for start, end in sorted(ranges):
    if merged and start <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
      merged.append((start, end))
  return merged


def overlap_length(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
  i = j = result = 0
  while i < len(left) and j < len(right):
    result += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
    if left[i][1] < right[j][1]:
      i += 1
    else:
      j += 1
  return result


def evaluate(
  gold_document: str, gold_ranges: list[tuple[int, int]], retrieved: list[TextSpan]
) -> RetrievalMetrics:
  """Score retrieved chunks, counting overlapping chunks only once."""
  gold = merge_ranges(gold_ranges)
  found = defaultdict(list)
  for chunk in retrieved:
    found[chunk.document_id].append((chunk.start, chunk.end))
  found = {name: merge_ranges(ranges) for name, ranges in found.items()}
  retrieved_chars = sum(
    end - start for ranges in found.values() for start, end in ranges
  )
  gold_chars = sum(end - start for start, end in gold)
  relevant_chars = overlap_length(found.get(gold_document, []), gold)
  return RetrievalMetrics(
    document_hit=gold_document in found,
    evidence_hit=relevant_chars > 0,
    precision=relevant_chars / retrieved_chars if retrieved_chars else 0.0,
    recall=relevant_chars / gold_chars,
  )
