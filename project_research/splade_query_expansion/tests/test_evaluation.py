"""評価指標の手計算例と、run読み込み時の境界条件を確認する。"""

import math
import tempfile
import unittest
from pathlib import Path

from evaluation import (
    evaluate_query,
    evaluate_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from run_evaluation import load_run


class EvaluationTests(unittest.TestCase):
    def test_graded_ndcg_uses_linear_gain(self) -> None:
        qrels = {"a": 3, "b": 1, "negative": -1}
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["b", "a"], qrels, 2), expected)
        self.assertEqual(ndcg_at_k(["a"], qrels, 1), 1)
        self.assertEqual(ndcg_at_k(["negative", "unjudged"], qrels), 0)

    def test_recall_and_reciprocal_rank_cutoffs(self) -> None:
        ranking = ["unjudged", "a", "b"]
        qrels = {"a": 1, "b": 2, "negative": 0}
        self.assertEqual(recall_at_k(ranking, qrels, 2), 0.5)
        self.assertEqual(recall_at_k(ranking, qrels, 100), 1)
        self.assertEqual(reciprocal_rank_at_k(ranking, qrels, 1), 0)
        self.assertEqual(reciprocal_rank_at_k(ranking, qrels, 2), 0.5)

    def test_default_metric_depths(self) -> None:
        ranking = [f"d{i}" for i in range(1000)]
        scores = evaluate_query(ranking, {"d9": 1, "d99": 1, "d999": 1})
        self.assertEqual(scores["recall@100"], 2 / 3)
        self.assertEqual(scores["recall@1000"], 1)
        self.assertEqual(scores["mrr@10"], 0.1)
        self.assertAlmostEqual(
            scores["ndcg@10"], (1 / math.log2(11)) / (1 + 1 / math.log2(3) + 0.5)
        )

    def test_missing_results_count_as_zero_and_no_positive_is_excluded(self) -> None:
        result = evaluate_run(
            {"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
        )
        self.assertEqual(result.excluded_query_ids, ["q3"])
        self.assertEqual(result.mean, dict.fromkeys(result.mean, 0.5))
        self.assertEqual(result.per_query["q2"], dict.fromkeys(result.mean, 0.0))

    def test_empty_rankings_and_invalid_inputs(self) -> None:
        self.assertEqual(ndcg_at_k([], {}), 0)
        self.assertEqual(recall_at_k([], {}), 0)
        self.assertEqual(reciprocal_rank_at_k([], {}), 0)
        for function in (ndcg_at_k, recall_at_k, reciprocal_rank_at_k):
            with self.assertRaises(ValueError):
                function(["a", "a"], {"a": 1})
            with self.assertRaises(ValueError):
                function([], {}, 0)
        with self.assertRaises(ValueError):
            evaluate_run({}, {"q": {"a": 0}})
        with self.assertRaises(ValueError):
            evaluate_run({}, {}, ["unknown"])
        with self.assertRaises(ValueError):
            evaluate_run({}, {"q": {"a": 1}}, ["q", "q"])

    def test_selected_subset_does_not_include_other_queries(self) -> None:
        result = evaluate_run({"q": ["a"]}, {"q": {"a": 1}, "other": {"b": 1}}, ["q"])
        self.assertEqual(result.mean["ndcg@10"], 1)
        self.assertEqual(list(result.per_query), ["q"])

    def test_run_preserves_explicit_rank_and_rejects_malformed_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.run"
            path.write_text("q Q0 b 2 1 tag\nq Q0 a 1 1 tag\n")
            self.assertEqual(load_run(path), {"q": ["a", "b"]})
            for text in (
                "q Q0 a 2 1 tag\n",
                "q Q0 a 1 2 tag\nq Q0 a 2 1 tag\n",
                "q Q0 a 1 nan tag\n",
                "q Q0 a 1 1 tag\nq Q0 b 2 2 tag\n",
                "invalid line\n",
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_run(path)
