"""SPLADEのクエリ推論と、語彙ごとの疎な重みの生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from splade_model import LoadedSplade

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class TokenWeight:
    """語彙ID・生のWordPiece表記・重みと、表示用のトークン分類。"""

    token_id: int
    token: str
    weight: float
    in_input: bool
    is_special: bool
    is_continuation: bool


@dataclass(frozen=True)
class SparseQuery:
    """クエリの全正重みトークンと、入力の切り詰め情報。"""

    query: str
    input_tokens: list[str]
    original_token_count: int
    used_token_count: int
    truncated: bool
    terms: list[TokenWeight]


def splade_max_pool(logits: Tensor, attention_mask: Tensor) -> Tensor:
    """MLMのlogitsからSPLADEの語彙重みを計算する。

    引数:
        logits: [バッチ数, 入力長, 語彙数]のMLM出力。確率へ変換する前の値。
        attention_mask: [バッチ数, 入力長]の有効位置。paddingは0とする。

    各値にReLUとlog(1+x)を適用し、padding位置を0にして入力位置方向の
    最大値を取る。戻り値は[バッチ数, 語彙数]の非負重み。NAVERのmax版
    SPLADEと同じ集約を使用する。softmaxや重み合計の正規化は行わない。
    """
    import torch

    activated = torch.log1p(torch.relu(logits))
    return (activated * attention_mask.unsqueeze(-1)).amax(dim=1)


def encode_query(loaded: LoadedSplade, query: str) -> SparseQuery:
    """自然文1件を推論し、SPLADEの正の重みを持つ全語彙を返す。

    引数:
        loaded: CPU上で評価モードにしたSPLADEモデルと対応トークナイザ。
        query: stemming前の自然文。空文字・空白のみの場合はValueError。

    勾配を計算せず、特殊トークンを含む入力長上限で切り詰めて推論する。
    重み降順、同点は語彙ID昇順に並べる。input_tokensはモデルに渡した
    WordPiece列、original_token_countは切り詰め前の特殊トークン込みの長さ。
    in_inputは実際の入力語彙IDとの一致であり、stemming後の元語判定ではない。

    戻り値:
        生のトークンとSPLADE重み、および入力・切り詰め情報。特殊トークンや
        継続部分語も重みが正なら残す。BM25索引語への変換、top-kによる
        拡張語選択、元語除外、重みの正規化は後続の処理で行う。
    """
    import torch

    if not query.strip():
        raise ValueError("クエリには空白以外の文字を指定してください")
    tokenizer = loaded.tokenizer
    original_ids = tokenizer.encode(query, truncation=False, verbose=False)
    inputs = tokenizer(
        query, return_tensors="pt", truncation=True, max_length=loaded.max_input_tokens
    )
    with torch.inference_mode():
        logits = loaded.model(**inputs).logits
        weights = splade_max_pool(logits, inputs["attention_mask"])[0]
    input_ids = inputs["input_ids"][0].tolist()
    input_id_set = set(input_ids)
    special_ids = set(tokenizer.all_special_ids)
    positive_ids = (weights > 0).nonzero(as_tuple=True)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(positive_ids)
    terms = [
        TokenWeight(
            token_id=token_id,
            token=token,
            weight=float(weights[token_id]),
            in_input=token_id in input_id_set,
            is_special=token_id in special_ids,
            is_continuation=token.startswith("##"),
        )
        for token_id, token in zip(positive_ids, tokens, strict=True)
    ]
    terms.sort(key=lambda term: (-term.weight, term.token_id))
    return SparseQuery(
        query=query,
        input_tokens=tokenizer.convert_ids_to_tokens(input_ids),
        original_token_count=len(original_ids),
        used_token_count=len(input_ids),
        truncated=len(original_ids) > len(input_ids),
        terms=terms,
    )
