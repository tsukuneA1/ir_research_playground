"""BM25の検索条件、順位付け、語ごとのスコア寄与。実行・保存処理はrun_bm25.py。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pyserini.analysis import Analyzer
    from pyserini.index.lucene import LuceneIndexReader
    from pyserini.search.lucene import LuceneSearcher


@dataclass(frozen=True)
class Hit:
    """検索結果の外部文書IDとBM25スコア。"""

    docid: str
    score: float


class TermContribution(TypedDict):
    """一つの索引語のクエリ内頻度と、文書スコアへの加点。"""

    term: str
    query_frequency: int
    contribution: float


class ScoreBreakdown(TypedDict):
    """検索APIのスコアと、索引語ごとに計算した寄与の内訳。"""

    docid: str
    search_score: float
    term_contributions: list[TermContribution]
    contribution_sum: float


def configure_bm25(searcher: LuceneSearcher, k1: float, b: float) -> Analyzer:
    """索引と同じPorter・既定ストップワード処理とBM25の係数を設定する。

    自然文クエリはPyseriniの既定BagOfWordsQueryGeneratorでOR検索する。
    同じ索引語がクエリに複数回出現した場合、その頻度を保持する。

    引数:
        searcher: 設定を変更する検索器。索引は事前に読み込んでおく。
        k1: 文書内の語の出現頻度に対する加点の飽和を制御する非負の係数。
        b: 文書長による補正の強さ。0では補正なし、1では最大になる。

    戻り値:
        検索器に設定したものと同じ前処理を、Pythonから確認するための
        アナライザ。呼び出し元でクエリの索引語を表示するときに使用する。
    """
    from pyserini.analysis import Analyzer, get_lucene_analyzer

    java_analyzer = get_lucene_analyzer(stemmer="porter", stopwords=True)
    searcher.set_analyzer(java_analyzer)
    searcher.set_bm25(k1, b)
    return Analyzer(java_analyzer)


def search_query(
    searcher: LuceneSearcher, query: str, qid: str, top_k: int
) -> list[Hit]:
    """クエリを検索し、自己一致と重複を除いて上位文書を返す。

    スコア降順、同点では外部文書IDの辞書順に並べる。取得境界に同点が
    残る場合は取得件数を増やし、その同点群を取得してから順位を確定する。
    これにより、境界付近の文書の採否がLuceneの内部文書IDに依存するのを
    避ける。除外後に件数が不足する場合も、追加で取得を試みる。

    引数:
        searcher: 前処理とBM25係数を設定済みの検索器。
        query: 前処理前の自然文クエリ。検索器側で索引語へ変換する。
        qid: クエリID。同じ外部文書IDを持つ自己一致文書を除外する。
        top_k: 除外後に返したい最大件数。正の整数を指定する。

    戻り値:
        順位順の検索結果。ヒットが少なければtop_k件未満、なければ空リスト。

    例外:
        ValueError: 検索APIが非有限のスコアを返した場合。
    """
    requested = min(searcher.num_docs, top_k + 1)
    if not requested:
        return []
    while True:
        raw = searcher.search(query, k=requested)
        scores: dict[str, float] = {}
        for hit in raw:
            docid, score = str(hit.docid), float(hit.score)
            if not math.isfinite(score):
                raise ValueError(f"非有限の検索スコア: {qid=}, {docid=}")
            if docid != str(qid):
                scores[docid] = max(scores.get(docid, -math.inf), score)
        ranked = sorted(
            (Hit(docid, score) for docid, score in scores.items()),
            key=lambda hit: (-hit.score, hit.docid),
        )
        exhausted = len(raw) < requested or requested >= searcher.num_docs
        cutoff_complete = (
            len(ranked) >= top_k and float(raw[-1].score) < ranked[top_k - 1].score
        )
        if exhausted or cutoff_complete:
            return ranked[:top_k]
        requested = min(searcher.num_docs, requested * 2)


def score_breakdown(
    reader: LuceneIndexReader,
    analyzer: Analyzer,
    query: str,
    hit: Hit,
    k1: float,
    b: float,
) -> ScoreBreakdown:
    """一つの文書のBM25スコアを、クエリの索引語ごとの寄与に分解する。

    クエリを検索時と同じアナライザで処理し、各索引語のBM25重みに
    クエリ内の出現回数を掛ける。処理済みの語にはanalyzer=Noneを指定し、
    stemmingを再適用しない。検索APIのスコアと寄与の合計には、丸めによる
    微小な差が生じ得る。

    引数:
        reader: 検索に使用した索引のリーダー。
        analyzer: 検索時と同じ前処理を行うアナライザ。
        query: 前処理前の自然文クエリ。
        hit: 内訳を確認する文書の外部IDと検索スコア。
        k1: 検索時と同じBM25の語頻度飽和係数。
        b: 検索時と同じBM25の文書長補正係数。

    戻り値:
        文書ID、検索スコア、索引語ごとの頻度と寄与、および寄与の合計。
    """
    terms = Counter(analyzer.analyze(query))
    contributions: list[TermContribution] = []
    for term, count in terms.items():
        weight = reader.compute_bm25_term_weight(
            hit.docid, term, analyzer=None, k1=k1, b=b
        )
        contributions.append(
            {"term": term, "query_frequency": count, "contribution": count * weight}
        )
    return {
        "docid": hit.docid,
        "search_score": hit.score,
        "term_contributions": contributions,
        "contribution_sum": sum(item["contribution"] for item in contributions),
    }
