"""データセットごとの索引・クエリ集合・検索時のIDの扱い。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """検索ロジックを共用するための、コーパスと評価クエリ集合の設定。"""

    name: str
    index: str
    collection: str
    field_description: str
    exclude_self_match: bool
    index_build_commit: str
    index_readme: str


DATASETS = {
    "scifact": DatasetSpec(
        name="scifact",
        index="beir-v1.0.0-scifact.flat",
        collection="beir-v1.0.0-scifact-test",
        field_description="contents (title + text)",
        exclude_self_match=True,
        index_build_commit="505594",
        index_readme=(
            "https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/blob/main/"
            "lucene-inverted/flat/lucene-inverted.beir-v1.0.0-flat.20221116.505594.README.md"
        ),
    ),
    "msmarco": DatasetSpec(
        name="msmarco",
        index="msmarco-v1-passage-full",
        collection="msmarco-v1-passage-dev",
        field_description="contents (original passage text, no added titles)",
        # クエリと文書は別のID体系。同じ数値でも自己一致とは限らない。
        exclude_self_match=False,
        index_build_commit="252b5e",
        index_readme=(
            "https://huggingface.co/datasets/castorini/prebuilt-indexes-msmarco-v1/blob/main/"
            "passage/original/lucene-inverted/tf/"
            "lucene-inverted.msmarco-v1-passage.20221004.252b5e.README.md"
        ),
    ),
}


def identify_dataset(index: str, collection: str) -> DatasetSpec:
    """索引とクエリ集合の両方が一致する設定を返し、未知の組はValueErrorにする。"""
    for dataset in DATASETS.values():
        if (dataset.index, dataset.collection) == (index, collection):
            return dataset
    raise ValueError(f"未対応の索引・クエリ集合です: {index}, {collection}")
