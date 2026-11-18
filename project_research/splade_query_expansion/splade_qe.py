"""SPLADE候補の索引語への変換と、重み付きBM25クエリの構築・検索。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from bm25_baseline import Hit, search_query
from splade_model import LoadedSplade
from splade_query import SparseQuery, encode_query

if TYPE_CHECKING:
    from pyserini.analysis import Analyzer
    from pyserini.index.lucene import LuceneIndexReader
    from pyserini.search.lucene import JQuery, LuceneSearcher

type Weighting = Literal["splade", "uniform"]


@dataclass(frozen=True)
class ExpansionTerm:
    """追加する索引語、最大重みの元トークン、SPLADE重みと正規化後の重み。"""

    term: str
    source_token: str
    raw_weight: float
    weight: float


@dataclass(frozen=True)
class ExpandedSearchResult:
    """疎表現、採用した追加語、Luceneクエリの表示と検索結果。"""

    sparse_query: SparseQuery
    expansion_terms: list[ExpansionTerm]
    lucene_query: str
    hits: list[Hit]


def select_expansion_terms(
    encoded: SparseQuery,
    analyzer: Analyzer,
    reader: LuceneIndexReader,
    k: int,
    weighting: Weighting = "splade",
) -> list[ExpansionTerm]:
    """設計書の順序で候補を選別し、有効な上位k索引語に合計1の重みを付ける。

    encodedは正規化前のSPLADE出力、analyzerとreaderは検索時と同じものを使う。
    特殊トークン・継続部分語・正でない重みを除き、候補をアナライザに通す。
    索引語がちょうど1個になる候補だけを残し、元クエリと同じ索引語や、
    索引に存在しない語を除く。元語判定はin_inputではなくstemming後に行う。

    同じ索引語の重みは最大値で集約する。重み降順・同点は索引語の辞書順で
    上位k個を取り、その後に重みを正規化する。uniformでも選択語は同じで、
    重みだけ1/実際の採用語数にする。候補がなければ空リストを返す。
    kの不正値、未知の重み方式、非有限の候補重みはValueErrorとする。
    """
    if k < 1:
        raise ValueError("追加語数kは正の整数で指定してください")
    if weighting not in ("splade", "uniform"):
        raise ValueError(f"未知の重み方式: {weighting}")
    original = set(analyzer.analyze(encoded.query))
    merged: dict[str, tuple[float, str]] = {}
    for candidate in encoded.terms:
        if not math.isfinite(candidate.weight):
            raise ValueError("SPLADE候補の重みが非有限です")
        if (
            candidate.weight <= 0
            or candidate.is_special
            or candidate.token.startswith("##")
        ):
            continue
        analyzed = analyzer.analyze(candidate.token)
        if len(analyzed) != 1:
            continue
        term = analyzed[0]
        if term in original:
            continue
        # analyzer=Noneはstemmingを行わない。処理済み索引語を再びstemしない。
        document_frequency, _ = reader.get_term_counts(term, analyzer=None)
        if not document_frequency:
            continue
        previous = merged.get(term)
        if previous is None or (-candidate.weight, candidate.token) < (
            -previous[0],
            previous[1],
        ):
            merged[term] = (candidate.weight, candidate.token)
    selected = sorted(merged.items(), key=lambda item: (-item[1][0], item[0]))[:k]
    total = sum(weight for _, (weight, _) in selected)
    return [
        ExpansionTerm(
            term=term,
            source_token=token,
            raw_weight=weight,
            weight=weight / total if weighting == "splade" else 1 / len(selected),
        )
        for term, (weight, token) in selected
    ]


def build_expanded_query(
    query: str,
    analyzer: Analyzer,
    expansion_terms: list[ExpansionTerm],
    expansion_weight: float,
) -> JQuery:
    """元BM25クエリと、λ×正規化重みでboostした追加語のORクエリを作る。

    queryは自然文、analyzerは検索時のアナライザ。元クエリはベースラインと
    同じBagOfWordsQueryGeneratorで構築し、反復語の頻度も保持する。
    追加索引語はTermQueryで直接渡して再stemmingを避ける。

    expansion_weightがλに対応し、有限の非負値を指定する。λ=0または追加語が
    空なら元クエリをそのまま返す。それ以外は元クエリと各追加語をSHOULDで
    結合し、全コーパスを検索する。追加語だけに一致する文書も候補になる。
    """
    from pyserini.pyclass import autoclass
    from pyserini.search.lucene import JBagOfWordsQueryGenerator

    if not math.isfinite(expansion_weight) or expansion_weight < 0:
        raise ValueError("λは有限の非負値で指定してください")
    original = JBagOfWordsQueryGenerator().buildQuery(
        "contents", analyzer.analyzer, query
    )
    if expansion_weight == 0 or not expansion_terms:
        return original
    builder = autoclass("org.apache.lucene.search.BooleanQuery$Builder")()
    should = autoclass("org.apache.lucene.search.BooleanClause$Occur").SHOULD
    term_class = autoclass("org.apache.lucene.index.Term")
    term_query_class = autoclass("org.apache.lucene.search.TermQuery")
    boost_query_class = autoclass("org.apache.lucene.search.BoostQuery")
    builder.add(original, should)
    for expansion in expansion_terms:
        term_query = term_query_class(term_class("contents", expansion.term))
        builder.add(
            boost_query_class(term_query, expansion_weight * expansion.weight), should
        )
    return builder.build()


def search_with_splade(
    loaded: LoadedSplade | None,
    searcher: LuceneSearcher,
    analyzer: Analyzer,
    query: str,
    qid: str,
    *,
    k: int,
    expansion_weight: float,
    weighting: Weighting = "splade",
    top_k: int = 1000,
    encoded: SparseQuery | None = None,
    exclude_self_match: bool = True,
) -> ExpandedSearchResult:
    """自然文1件のSPLADE推論、追加語選別・重み付け、全コーパス検索を行う。

    loadedはCPU上のSPLADEモデル、searcherとanalyzerは設定済みのBM25検索器と
    前処理。kは追加語数、expansion_weightはλ、top_kは取得文書数を指定する。
    weightingはspladeまたはuniformを指定し、選択する語は両者で共通にする。

    キャッシュ済みのencodedを渡すと推論を省略でき、その場合だけloadedは
    Noneでもよい。原文が異なるキャッシュはValueErrorとする。
    戻り値には全正重みの疎表現、選択した語と重み、検索結果を含める。
    既存BM25の上位候補に限定した再順位付けではなく、重み付きクエリで
    再検索する。自己一致除外・件数補充・同点処理はBM25と同じ関数を使う。
    クエリと文書のID体系が異なるMS MARCOではexclude_self_match=Falseを渡す。
    """
    if encoded is None:
        if loaded is None:
            raise ValueError("未推論クエリにはSPLADEモデルが必要です")
        encoded = encode_query(loaded, query)
    if encoded.query != query:
        raise ValueError("SPLADE出力と検索クエリの原文が異なります")
    terms = select_expansion_terms(
        encoded, analyzer, searcher.index_reader, k, weighting
    )
    lucene_query = build_expanded_query(query, analyzer, terms, expansion_weight)
    hits = search_query(
        searcher, lucene_query, qid, top_k, exclude_self_match=exclude_self_match
    )
    return ExpandedSearchResult(encoded, terms, lucene_query.toString(), hits)
