# LegalBench-RAGでBM25を試す

契約文書の検索を理解するための、Python標準ライブラリだけで動く最小の実験です。LegalBench-RAGの[ContractNLI部分](https://huggingface.co/datasets/awinml/legalbench-rag)を使い、質問から関連する契約の根拠箇所を探します。元の[LegalBench-RAG論文](https://arxiv.org/abs/2408.10343)と[公開コード](https://github.com/ZeroEntropy-AI/legalbenchrag)も参照してください。

## 動かし方

Python 3.10以降と`uv`を使用します。実験本体に追加パッケージは不要です。Ruffは開発用依存として`uv`が用意します。

```bash
uv sync
uv run python download_contractnli.py
uv run python bm25_baseline.py > results/contractnli_bm25.json
```

`results`ディレクトリはこのリポジトリに用意しています。取得した文書と評価用JSONは`data/`に保存され、Gitには含めません。ダウンロード元はHugging Face上のLegalBench-RAGミラーで、リビジョンを固定し、配布物のSHA-256で各ファイルを検証します。元のDropboxリンクは2026-10-19時点で無効でした。

手元のデータがあれば、`uv run python bm25_baseline.py --data-dir /path/to/data`でも実行できます。その場合は`benchmarks/contractnli.json`と`corpus/contractnli/*.txt`の構成にしてください。

整形は`uv run ruff format .`、整形状態の確認は`uv run ruff format --check .`です。静的チェックは`uv run ruff check .`で実行できます。`pyproject.toml`でインデント幅をスペース2文字に固定しています。

## コードの読み方

| ファイル | 役割 |
| --- | --- |
| [`bm25.py`](bm25.py) | 単語への分割、チャンク作成、転置索引、BM25のスコア計算 |
| [`evaluation.py`](evaluation.py) | 重複する文字範囲の結合と、根拠箇所に対する評価 |
| [`legalbenchrag_io.py`](legalbenchrag_io.py) | LegalBench-RAG固有のJSONとファイルを読み、`Question`に変換・検証 |
| [`bm25_baseline.py`](bm25_baseline.py) | 上記を組み合わせて実験し、結果をJSONで出力 |
| [`download_contractnli.py`](download_contractnli.py) | 配布データの取得とチェックサム検証 |

BM25の仕組みを追うなら`bm25.py`から読めます。データセットのJSONキー名は`legalbenchrag_io.py`の中だけで扱います。

## 何をしているか

1. 95契約を単語200個ごと、40単語重複で分割します。各分割片は原文での開始・終了位置を保持します。
2. 各分割片にBM25の索引を作り、977問それぞれの上位5件を検索します。語は英数字とアポストロフィで区切り、小文字にします。BM25の`k1=1.2, b=0.75`です。
3. **all_contracts**では全95契約から、**known_contract**では正解契約の中だけから検索します。後者は原因を調べるための診断条件で、実運用で正解契約が既知という意味ではありません。
4. 取得した分割片と正解根拠の文字範囲が重なった割合を計算します。重複した分割片は文字範囲を結合し、同じ文字を二重に数えません。

指標は質問ごとの平均です。`document_hit`は上位5件に正解契約がある割合、`evidence_hit`は上位5件のどれかが正解根拠に1文字以上触れた割合です。`recall`は正解根拠の文字のうち取得できた割合、`precision`は取得した文字のうち正解根拠に当たる割合です。検索した分割片は約200単語と長いため、precisionの値は低くなりやすい点に注意してください。

## 正解ラベルと指標の意味

[元のContractNLI](https://aclanthology.org/2021.findings-emnlp.164/)では、契約と仮説の組に対し「含意・矛盾・言及なし」の判定と、その判断の根拠となる文中の範囲が注釈されています。LegalBench-RAGはその根拠を`file_path`と半開区間の文字位置`span: [start, end]`に対応付けています。この実験のJSONにある正解は**質問ごとの根拠範囲**であり、チャンクごとの0/1や段階的な関連度ラベルではありません。各質問には少なくとも1つの正解範囲があります。

このため、現在の`precision`と`recall`は**文字単位の根拠一致率**です。`precision = 取得文字と正解範囲の重なり / 取得文字数`、`recall = 同じ重なり / 正解範囲の文字数`として計算します。`document_hit`と`evidence_hit`も出しています。たとえば正解が100文字で、取得した1,000文字のうち80文字が正解と重なれば、precisionは0.08、recallは0.80です。

MRRやnDCG@kも、チャンクと正解範囲の重なりから**チャンク単位の関連度を新たに定義すれば**計算できます。ただし、それはデータセットに元からある専門家のチャンク関連度ラベルではありません。特に「1文字でも重なれば関連あり」では分割境界に結果が左右されるため、このベースラインではまず元の根拠範囲を直接使う指標を報告します。正解範囲外の有用な説明も、現在の厳密一致のprecisionでは得点になりません。

## 実行結果

2026-10-19に`uv run python bm25_baseline.py`の既定値で実行した結果です。全質問の正解箇所について、原文からの文字位置を照合してから評価しています。

| 検索範囲 | document_hit | evidence_hit | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| 全95契約 | 0.4176 | 0.1863 | 0.0097 | 0.1407 |
| 正解契約のみ | 1.0000 | 0.8721 | 0.0604 | 0.7900 |

索引には989分割片が入りました。今回の条件では正解契約が分かればBM25が根拠に触れる割合は高く、全契約からでは大きく落ちます。これは次の実験で、**文書特定**と**文書内の根拠検索**を分けて比較する理由になります。この1回の結果だけで他手法の優劣や一般的な法令検索への性能は判断できません。

結果JSONには検索上位5件の例も入ります。別の質問を表示するには`--show-query 10`のように質問番号を指定します。分割や件数は`--chunk-size 200 --overlap 40 --top-k 5`で変更できます。評価計算の小さな確認は`uv run python -m unittest test_bm25_baseline.py`で実行します。
