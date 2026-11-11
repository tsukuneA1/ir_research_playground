from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index(
    "beir-v1.0.0-scifact.flat"
)

topics = get_topics("beir-v1.0.0-scifact-test")
qrels = get_qrels("beir-v1.0.0-scifact-test")

print(topics, qrels)
