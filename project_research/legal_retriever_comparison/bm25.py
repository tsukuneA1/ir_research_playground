"""Dataset-independent token chunking and BM25 retrieval."""

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class Chunk:
  document_id: str
  start: int
  end: int
  terms: Counter[str]


def make_chunks(document_id: str, content: str, size: int, overlap: int) -> list[Chunk]:
  """Split by word tokens while retaining character offsets into the original text."""
  if size <= 0 or not 0 <= overlap < size:
    raise ValueError("Require size > overlap >= 0")
  matches = list(TOKEN.finditer(content))
  step = size - overlap
  chunks = []
  for first in range(0, len(matches), step):
    part = matches[first : first + size]
    if not part:
      break
    chunks.append(
      Chunk(
        document_id,
        part[0].start(),
        part[-1].end(),
        Counter(match.group().lower() for match in part),
      )
    )
    if first + size >= len(matches):
      break
  return chunks


class BM25:
  """Inverted index over chunks using Robertson-style BM25 scores."""

  def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
    if not chunks:
      raise ValueError("BM25 needs at least one chunk")
    self.chunks = chunks
    self.k1 = k1
    self.b = b
    self.lengths = [sum(chunk.terms.values()) for chunk in chunks]
    self.avg_length = sum(self.lengths) / len(chunks)
    self.postings = defaultdict(list)
    for index, chunk in enumerate(chunks):
      for term, count in chunk.terms.items():
        self.postings[term].append((index, count))

  def search(
    self, query: str, top_k: int, allowed_document: str | None = None
  ) -> list[tuple[int, float]]:
    """Return (chunk index, score), highest first. The filter is diagnostic."""
    scores = defaultdict(float)
    total = len(self.chunks)
    for term in sorted({match.group().lower() for match in TOKEN.finditer(query)}):
      posting = self.postings.get(term, [])
      if not posting:
        continue
      idf = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
      for index, tf in posting:
        if (
          allowed_document is not None
          and self.chunks[index].document_id != allowed_document
        ):
          continue
        normalizer = self.k1 * (
          1 - self.b + self.b * self.lengths[index] / self.avg_length
        )
        scores[index] += idf * tf * (self.k1 + 1) / (tf + normalizer)
    return heapq.nlargest(top_k, scores.items(), key=lambda pair: (pair[1], -pair[0]))
