"""順位付き文書IDとqrelsから評価指標を計算する。ファイル入出力は行わない。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from statistics import fmean

type Judgments = Mapping[str, int]
type Qrels = Mapping[str, Judgments]
type Rankings = Mapping[str, Sequence[str]]
type MetricScores = dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    """クエリ別の指標、算術平均、正の関連文書がなく平均から除外したID。"""

    mean: MetricScores
    per_query: dict[str, MetricScores]
    excluded_query_ids: list[str]


def _validate_ranking(ranking: Sequence[str], k: int) -> None:
    """正のカットオフと、重複のない順位付き文書IDであることを確認する。"""
    if k < 1:
        raise ValueError("カットオフkは正の整数で指定してください")
    if len(ranking) != len(set(ranking)):
        raise ValueError("ランキング内の文書IDが重複しています")


def ndcg_at_k(ranking: Sequence[str], qrels: Judgments, k: int = 10) -> float:
    """上位k件のnDCGを計算する。関連度をそのまま利得とする線形利得を使う。

    rankingは順位順の文書ID、qrelsは文書IDと整数の関連度の対応。
    DCGは各関連度をlog2(順位+1)で割って加算し、qrels全体から作った
    理想順位のDCGで正規化する。trec_evalのndcg_cutと同じ利得の定義を使い、
    2の関連度乗から1を引く方式は使わない。
    未判定文書と関連度0以下の文書の利得は0とする。空のランキングや
    正の関連文書がない場合は0を返す。文書IDの重複やkの不正値はValueError。
    """
    _validate_ranking(ranking, k)
    dcg = sum(
        max(qrels.get(docid, 0), 0) / log2(rank + 1)
        for rank, docid in enumerate(ranking[:k], 1)
    )
    ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)
    ideal_dcg = sum(grade / log2(rank + 1) for rank, grade in enumerate(ideal[:k], 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def recall_at_k(ranking: Sequence[str], qrels: Judgments, k: int = 100) -> float:
    """上位k件に含まれる既知の関連文書数を、既知の関連文書総数で割る。

    rankingは順位順の文書ID、qrelsは文書IDと整数の関連度の対応。
    正の関連度を持つ文書を関連ありとする。未判定文書は関連数に含めない。
    分母はqrels上の正例数であり、実在する関連文書の総数とは限らない。
    空のランキングや正例がない場合は0、文書IDの重複やkの不正値はValueError。
    """
    _validate_ranking(ranking, k)
    relevant = {docid for docid, grade in qrels.items() if grade > 0}
    return len(relevant.intersection(ranking[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank_at_k(
    ranking: Sequence[str], qrels: Judgments, k: int = 10
) -> float:
    """上位k件で最初に現れる関連文書の順位の逆数を返す。

    rankingは順位順の文書ID、qrelsは文書IDと整数の関連度の対応。
    正の関連度を持つ文書を関連ありとし、上位k件に存在しなければ0を返す。
    この値は1クエリのRRであり、クエリ間で算術平均したものがMRRになる。
    文書IDの重複やkの不正値はValueErrorとする。
    """
    _validate_ranking(ranking, k)
    for rank, docid in enumerate(ranking[:k], 1):
        if qrels.get(docid, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate_query(ranking: Sequence[str], qrels: Judgments) -> MetricScores:
    """1クエリのnDCG@10、Recall@100・1000、RR@10を計算する。

    rankingとqrelsの文書IDは文字列に統一して渡す。検索結果の順序を
    そのまま使用し、スコアによる並べ替えや自己一致除外は行わない。
    戻り値のmrr@10には、このクエリのRR@10を入れる。平均時にMRR@10になる。
    """
    return {
        "ndcg@10": ndcg_at_k(ranking, qrels, 10),
        "recall@100": recall_at_k(ranking, qrels, 100),
        "recall@1000": recall_at_k(ranking, qrels, 1000),
        "mrr@10": reciprocal_rank_at_k(ranking, qrels, 10),
    }


def evaluate_run(
    rankings: Rankings, qrels: Qrels, query_ids: Sequence[str] | None = None
) -> EvaluationResult:
    """対象クエリ別に指標を計算し、クエリ間の算術平均を返す。

    rankingsはクエリIDと順位順の文書ID列、qrelsはクエリIDごとの関連度判定。
    query_idsを省略するとqrelsの全クエリを評価する。結果のない対象クエリも
    空ランキングとして0点にし、平均から落とさない。qrelsに正例がない
    クエリは平均から除外してexcluded_query_idsに記録する。
    選択されていないクエリのランキングは平均に含めない。
    対象IDの重複、qrelsにない対象ID、評価可能なクエリが0件ならValueError。
    """
    selected = list(query_ids) if query_ids is not None else sorted(qrels)
    if len(selected) != len(set(selected)):
        raise ValueError("評価対象のクエリIDが重複しています")
    missing = set(selected) - qrels.keys()
    if missing:
        raise ValueError(f"qrelsにない評価対象クエリです: {sorted(missing)}")
    excluded = [qid for qid in selected if not any(g > 0 for g in qrels[qid].values())]
    per_query = {
        qid: evaluate_query(rankings.get(qid, ()), qrels[qid])
        for qid in selected
        if any(grade > 0 for grade in qrels[qid].values())
    }
    if not per_query:
        raise ValueError("正の関連文書を持つ評価対象クエリがありません")
    names = next(iter(per_query.values())).keys()
    mean = {
        name: fmean(scores[name] for scores in per_query.values()) for name in names
    }
    return EvaluationResult(mean, per_query, excluded)
