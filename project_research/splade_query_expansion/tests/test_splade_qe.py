"""小さな候補集合で設計書の選別順序・集約・重み付けを確認する。"""

import unittest

from splade_qe import select_expansion_terms
from splade_query import SparseQuery, TokenWeight


class Analyzer:
    def analyze(self, text: str) -> list[str]:
        return {
            "query": ["lung"],
            "lungs": ["lung"],
            "alpha": ["a"],
            "alphas": ["a"],
            "beta": ["b"],
            "fragment": ["fragment"],
            "missing": ["missing"],
            "split": ["two", "terms"],
            "stop": [],
        }[text]


class Reader:
<<<<<<< Updated upstream
=======
    def stats(self) -> dict[str, int]:
        return {"documents": 100}

>>>>>>> Stashed changes
    def get_term_counts(self, term: str, analyzer=None) -> tuple[int, int]:
        return (0, 0) if term == "missing" else (1, 1)


class SelectionTests(unittest.TestCase):
<<<<<<< Updated upstream
=======
    def test_df_filter_before_top_k_and_normalization(self) -> None:
        class FrequencyReader(Reader):
            def get_term_counts(self, term, analyzer=None):
                return {"a": (10, 10), "b": (9, 9), "fragment": (1, 1)}.get(
                    term, (0, 0)
                )

        reader = FrequencyReader()
        unfiltered = select_expansion_terms(self.encoded, Analyzer(), reader, 2)
        filtered = select_expansion_terms(
            self.encoded, Analyzer(), reader, 2, max_df_ratio=0.1
        )
        self.assertEqual([term.term for term in unfiltered], ["a", "b"])
        self.assertEqual([term.term for term in filtered], ["b", "fragment"])
        self.assertAlmostEqual(filtered[0].weight, 8 / 11)
        self.assertAlmostEqual(sum(term.weight for term in filtered), 1)
        uniform = select_expansion_terms(
            self.encoded, Analyzer(), reader, 2, "uniform", max_df_ratio=0.1
        )
        self.assertEqual([term.weight for term in uniform], [0.5, 0.5])
        self.assertEqual(
            select_expansion_terms(
                self.encoded, Analyzer(), reader, 2, max_df_ratio=0.01
            ),
            [],
        )

    def test_df_filter_rejects_invalid_threshold_and_empty_index(self) -> None:
        for ratio in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                select_expansion_terms(
                    self.encoded, Analyzer(), Reader(), 5, max_df_ratio=ratio
                )

        class EmptyReader(Reader):
            def stats(self):
                return {"documents": 0}

        with self.assertRaises(ValueError):
            select_expansion_terms(
                self.encoded, Analyzer(), EmptyReader(), 5, max_df_ratio=0.1
            )

>>>>>>> Stashed changes
    def setUp(self) -> None:
        self.encoded = SparseQuery(
            "query",
            [],
            1,
            1,
            False,
            [
                TokenWeight(1, "[SEP]", 100, True, True, False),
                TokenWeight(2, "##ing", 100, False, False, True),
                TokenWeight(3, "lungs", 100, False, False, False),
                TokenWeight(4, "missing", 100, False, False, False),
                TokenWeight(5, "split", 100, False, False, False),
                TokenWeight(6, "stop", 100, False, False, False),
                TokenWeight(7, "alpha", 6, False, False, False),
                TokenWeight(8, "alphas", 8, False, False, False),
                TokenWeight(9, "beta", 8, False, False, False),
                # モデルの入力断片でも、元クエリの索引語と異なれば選択可能。
                TokenWeight(10, "fragment", 3, True, False, False),
            ],
        )

    def test_filter_merge_then_cutoff(self) -> None:
        terms = select_expansion_terms(self.encoded, Analyzer(), Reader(), 1)
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].term, "a")
        self.assertEqual(terms[0].source_token, "alphas")
        self.assertEqual(terms[0].raw_weight, 8)
        self.assertEqual(terms[0].weight, 1)

    def test_uniform_uses_same_terms_and_total_weight(self) -> None:
        splade = select_expansion_terms(self.encoded, Analyzer(), Reader(), 50)
        uniform = select_expansion_terms(
            self.encoded, Analyzer(), Reader(), 50, "uniform"
        )
        self.assertEqual([term.term for term in splade], ["a", "b", "fragment"])
        self.assertEqual(
            [term.term for term in uniform], [term.term for term in splade]
        )
        self.assertAlmostEqual(sum(term.weight for term in splade), 1)
        self.assertAlmostEqual(sum(term.weight for term in uniform), 1)
        self.assertEqual(splade[0].weight, 8 / 19)
        self.assertTrue(all(term.weight == 1 / 3 for term in uniform))

    def test_empty_and_invalid_settings(self) -> None:
        empty = SparseQuery("query", [], 1, 1, False, [])
        self.assertEqual(select_expansion_terms(empty, Analyzer(), Reader(), 5), [])
        with self.assertRaises(ValueError):
            select_expansion_terms(empty, Analyzer(), Reader(), 0)
        with self.assertRaises(ValueError):
            select_expansion_terms(empty, Analyzer(), Reader(), 5, "unknown")
