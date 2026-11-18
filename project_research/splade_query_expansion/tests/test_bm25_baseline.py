"""Small ranking contract checks; no network, JVM, or index download."""

import unittest
from types import SimpleNamespace

from bm25_baseline import Hit, search_query
from retrieval_datasets import DATASETS, identify_dataset
from run_bm25 import format_run, normalize_data


class Searcher:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self.hits = [SimpleNamespace(docid=docid, score=score) for docid, score in hits]
        self.num_docs = len(hits)
        self.calls: list[int] = []

    def search(self, query: str, k: int) -> list[SimpleNamespace]:
        self.calls.append(k)
        return self.hits[:k]


class RankingTests(unittest.TestCase):
    def test_msmarco_numeric_id_collision_is_not_a_self_match(self) -> None:
        dataset = DATASETS["msmarco"]
        self.assertEqual(
            search_query(
                Searcher([("42", 3), ("a", 2)]),
                "query",
                "42",
                1,
                exclude_self_match=dataset.exclude_self_match,
            ),
            [Hit("42", 3)],
        )

    def test_dataset_requires_matching_index_and_collection(self) -> None:
        dataset = DATASETS["msmarco"]
        self.assertEqual(identify_dataset(dataset.index, dataset.collection), dataset)
        with self.assertRaises(ValueError):
            identify_dataset(dataset.index, DATASETS["scifact"].collection)

    def test_complete_cutoff_tie_before_lexical_sort(self) -> None:
        searcher = Searcher([("z", 3), ("y", 2), ("x", 2), ("a", 2), ("b", 1)])
        hits = search_query(searcher, "query", "q", 2)
        self.assertEqual(hits, [Hit("z", 3), Hit("a", 2)])
        self.assertGreater(len(searcher.calls), 1)

    def test_refill_after_self_and_duplicates(self) -> None:
        searcher = Searcher([("q", 5), ("a", 4), ("a", 4), ("b", 3), ("c", 2)])
        self.assertEqual(
            search_query(searcher, "query", "q", 2), [Hit("a", 4), Hit("b", 3)]
        )

    def test_zero_and_short_results(self) -> None:
        self.assertEqual(search_query(Searcher([]), "query", "q", 10), [])
        self.assertEqual(
            search_query(Searcher([("a", 1)]), "query", "q", 10), [Hit("a", 1)]
        )

    def test_normalize_ids_and_grades_and_detect_missing_query(self) -> None:
        queries, qrels = normalize_data({1: {"title": "query"}}, {1: {42: "1"}})
        self.assertEqual(queries, {"1": "query"})
        self.assertEqual(qrels, {"1": {"42": 1}})
        with self.assertRaises(ValueError):
            normalize_data({}, {1: {42: "1"}})

    def test_run_format(self) -> None:
        self.assertEqual(
            format_run("q", [Hit("a", 2), Hit("b", 1)]),
            "q Q0 a 1 2 bm25-scifact\nq Q0 b 2 1 bm25-scifact\n",
        )


if __name__ == "__main__":
    unittest.main()
