"""モデルを取得せず、小さなテンソルでSPLADEの集約を確認する。"""

import math
import unittest

import torch

from splade_query import splade_max_pool


class SpladePoolingTests(unittest.TestCase):
    def test_relu_log_max_and_padding_mask(self) -> None:
        logits = torch.tensor(
            [[[0.0, -1.0, -2.0], [2.0, 4.0, -1.0], [99.0, 99.0, 99.0]]]
        )
        weights = splade_max_pool(logits, torch.tensor([[1, 1, 0]]))
        expected = torch.tensor([[math.log(3), math.log(5), 0]])
        torch.testing.assert_close(weights, expected)

    def test_all_padding_has_no_weight(self) -> None:
        logits = torch.tensor([[[2.0, 4.0]]])
        weights = splade_max_pool(logits, torch.tensor([[0]]))
        torch.testing.assert_close(weights, torch.zeros(1, 2))
