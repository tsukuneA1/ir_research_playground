# MS MARCO v1 passageでの実行

このディレクトリで実行する。SciFactと同じBM25・SPLADE QE・評価処理を使い、
`--dataset msmarco`で索引とクエリ集合、結果・キャッシュの既定保存先を切り替える。
省略時のBM25は従来どおりSciFactを使う。

## 索引とクエリ集合

- 索引: `msmarco-v1-passage-full`（8,841,823 passage）。
- クエリ・qrels: `msmarco-v1-passage-dev`（6,980クエリ）。
- `full`は文書ベクトルを含み、今後のRM3との比較にも使える。
- 初回はPyseriniが約3.7 GBの配布アーカイブを取得・展開する。索引は全コーパス共通で、少数クエリでも同じ容量が必要。
- クエリIDと文書IDは別の体系なので、数値が一致する文書も検索結果に残す。

## まず3クエリで確認

```bash
uv run python run_bm25.py --dataset msmarco --limit 3 \
  --output results/msmarco/smoke-bm25.run
uv run python run_splade_qe.py --dataset msmarco \
  --baseline-metadata results/msmarco/smoke-bm25.run.json \
  --k 5 --lambda 1 --output results/msmarco/smoke-splade.run
uv run python run_evaluation.py --run results/msmarco/smoke-splade.run
```

`--limit`は文字列としてソートしたIDの先頭から取る動作確認用の指定で、
無作為抽出ではない。ここでのk・λも動作確認用で、調整済みの値ではない。
評価CLIはnDCG@10・Recall@100・Recall@1000・MRR@10を保存する。
この少数件の値からデータセット全体の性能は判断しない。

## 対象クエリを指定する

BM25の`--query-id`を複数回指定するか、`--query-ids-file`で文字列IDのJSON配列を渡せる。
例えば`["1000000", "1000004"]`を保存したファイルを使う場合:

```bash
uv run python run_bm25.py --dataset msmarco --query-ids-file query_ids.json \
  --output results/msmarco/subset-bm25.run
uv run python run_splade_qe.py \
  --baseline-metadata results/msmarco/subset-bm25.run.json \
  --k 5 --lambda 1 --output results/msmarco/subset-splade.run
```

QEはベースラインのメタデータからデータセットと対象クエリを引き継ぐ。
`--weighting uniform`で同じ追加語の均等重み版を実行できる。
推論キャッシュは既定で`results/msmarco/splade-cache/`に置き、k・λ・重み方式を変えても再利用する。

全6,980クエリを実行するときはBM25のクエリ指定と`--limit`を省略する。
パラメータ調整用と最終評価用のクエリ分割・分割比率はこの変更では決定していない。
