import unittest
from collections import Counter

from bm25 import BM25, Chunk, make_chunks
from evaluation import evaluate, merge_ranges, overlap_length


class BaselineTests(unittest.TestCase):
  def test_bm25_prefers_matching_chunk_and_respects_file_filter(self):
    chunks = [
      Chunk("a", 0, 12, Counter({"confidential": 1, "information": 1})),
      Chunk("b", 0, 10, Counter({"termination": 1, "notice": 1})),
    ]
    index = BM25(chunks)
    self.assertEqual(index.search("confidential information", 1)[0][0], 0)
    self.assertEqual(index.search("confidential information", 1, "b"), [])

  def test_overlap_merges_repeated_chunks_and_ignores_wrong_file(self):
    chunks = [
      Chunk("a", 0, 10, Counter()),
      Chunk("a", 8, 18, Counter()),
      Chunk("b", 0, 10, Counter()),
    ]
    result = evaluate("a", [(5, 15)], chunks)
    self.assertAlmostEqual(result.recall, 1.0)
    self.assertAlmostEqual(result.precision, 10 / 28)
    self.assertEqual(merge_ranges([(0, 10), (8, 18)]), [(0, 18)])
    self.assertEqual(overlap_length([(0, 18)], [(5, 15)]), 10)

  def test_chunk_offsets_point_into_original_text(self):
    content = "First clause.\nSecond clause; third clause."
    chunks = make_chunks("a", content, size=3, overlap=1)
    self.assertEqual(content[chunks[0].start : chunks[0].end], "First clause.\nSecond")
    self.assertEqual(content[chunks[1].start : chunks[1].end], "Second clause; third")


if __name__ == "__main__":
  unittest.main()
