# SPLADEの追加語だけをBM25に使う実験

問い: **SPLADEが選んだクエリ追加語は、SPLADEのスコアを捨ててもBM25の検索精度を上げるか。**

悪化した質問の追加語、順位、BM25加点を追った結果は[失敗例の分析](failure_analysis.md)にまとめています。
追加語の加点を弱める実験とzero-hit fallbackの判定結果は[重みとfallbackの分析](weight_analysis.md)にまとめています。
BEIRで報告されたクエリ・文書両側のSPLADEとの数値比較は[既存研究との比較](literature_comparison.md)にまとめています。

このディレクトリは `legal_retriever_comparison` から独立しています。[BEIR](https://github.com/beir-cellar/beir) の **SciFact**（科学文献、5,183文書・test 300問）、**FiQA**（金融、57,638文書・test 648問）、**ArguAna**（議論、8,674文書・test 1,406問）、**NFCorpus**（医療・栄養、3,633文書・test 323問）を使います。文書はタイトルと本文を連結し、小文字の英数字列でトークン化します。BM25は `k1=1.2, b=0.75`。SPLADEは質問だけに適用し、元の質問にない上位語をBM25へ追加します。文書の拡張、SPLADEの語の重み、SPLADE検索スコアとの融合は使いません。

## 実行

Python 3.12と `uv` を使います。公開モデルをローカルCPUで実行するので、API料金はかかりません。モデル本体のダウンロードは約438 MB、CPU版PyTorchは約170 MBです。メモリ使用量を抑えるため推論スレッド数を2にしています。

```bash
cd project_research/splade_query_expansion
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare.py
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python experiment.py

# 別ドメインのFiQAを実行
.venv/bin/python prepare.py --dataset fiqa
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python experiment.py --dataset fiqa
.venv/bin/python compare_prefixes.py --dataset fiqa --expansions results/fiqa/expansions_3terms_min2.json --output results/fiqa/recall_cutoffs.json

# 新しい2ドメイン。データ取得と推論は1つずつ実行する
.venv/bin/python prepare.py --dataset arguana
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python experiment.py --dataset arguana
.venv/bin/python prepare.py --dataset nfcorpus
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python experiment.py --dataset nfcorpus
.venv/bin/python weight_and_fallback.py --dataset arguana --expansions results/arguana/expansions_3terms_min2.json --output results/arguana/weights.json
.venv/bin/python weight_and_fallback.py --dataset nfcorpus --expansions results/nfcorpus/expansions_3terms_min2.json --output results/nfcorpus/weights.json
```

`prepare.py` は各zipをSHA-256で確認し、[公開SPLADEモデル](https://huggingface.co/naver/splade-cocondenser-ensembledistil)をコミット `49cf4c7b0db5b870a401ddf5e2669993ef3699c7` に固定して取得します。データとモデルは `data/`、詳細結果と追加語は `results/` に保存し、Gitには含めません。`compare_prefixes.py` は保存済みの3語を使い、索引を1回作るだけで0～3語の条件を再計算します。

## 条件と結果

2026-10-21にこのPCで実行した値です。主指標はnDCG@10、補助指標はRecall@10です。語はSPLADE重み順に選びますが、BM25ではすべて同じ重みです。元の質問語は必ず残します。追加語は単独の英字語、2文字以上、BM25の索引に存在する語に限定しました。

### SciFact

| 条件 | nDCG@10 | Recall@10 | nDCG改善/悪化/同点の質問数 |
| --- | ---: | ---: | ---: |
| BM25のみ | 0.66046 | 0.78761 | — |
| SPLADE上位1語を追加 | 0.66435 | 0.79178 | 12 / 19 / 269 |
| SPLADE上位2語を追加 | 0.65426 | 0.78289 | 21 / 36 / 243 |
| SPLADE上位3語を追加 | 0.65337 | 0.78578 | 27 / 42 / 231 |
| 上位3語、4文字以上 | 0.64233 | 0.78211 | 29 / 53 / 218 |

3語追加を最初の条件として実行しました。その後、短い語の混入を見て4文字以上の条件と1・2語の条件を探索的に追加しました。1語追加の改善幅は小さく、質問単位の差を復元抽出した95%区間はおよそ `[-0.0076, +0.0164]` で0を含みます。**この1回の結果から改善を断定できません。** 3語追加も区間はおよそ `[-0.0256, +0.0102]` です。

例: 質問54では `lungs` に対して `lung` が追加され、nDCG@10が0から1へ改善しました。質問660では `Ivermectin` に対して `iv` が追加され、1から0.5へ悪化しました。追加語には同義語だけでなく、語形変化、関連語、略語や短い断片も含まれます。`lung` の例は、単純な語幹処理でも改善し得るため、SPLADE固有の意味理解の証拠にはなりません。

比較結果は `results/comparison.json`（3語）、`comparison_1term.json`、`comparison_2terms.json`、`comparison_min4.json` にあります。各ファイルの `per_query` に質問、追加語、条件ごとのスコアを保存しています。

```bash
# 保存済みの追加語を使い、推論せずに1語条件を再計算する
.venv/bin/python experiment.py --terms 1 --expansions results/expansions.json --output results/comparison_1term.json
```

このBM25は実験用の単純な実装で、ステミングをせず、LuceneやPyseriniと同じスコア値になることは意図していません。比較の両条件では同じ索引と評価方法を使います。SciFactの未判定文書は非関連として扱います。モデルはMS MARCO系で学習された英語モデルなので、科学文献での結果が他の分野にそのまま移るわけではありません。

### FiQA

同じモデルとBM25設定で[BEIR FiQA](https://huggingface.co/datasets/BeIR/fiqa)のtest全648問を評価しました。1～3語条件は同じ上位3語の先頭から取っています。

| 条件 | nDCG@10 | Recall@10 |
| --- | ---: | ---: |
| BM25のみ | 0.23585 | 0.29823 |
| SPLADE上位1語を追加 | 0.22858 | 0.29411 |
| SPLADE上位2語を追加 | 0.22392 | 0.29503 |
| SPLADE上位3語を追加 | 0.21704 | 0.28161 |

FiQAでは語数を増やすほどnDCG@10が下がりました。3語条件は648問中95問で改善、164問で悪化、389問で同点でした。質問単位の差を5,000回復元抽出したnDCG@10差の95%区間はおよそ `[-0.0317, -0.0055]` です。SciFactで見えた1語の微増はFiQAでは再現しませんでした。

質問11088では `wills, lawyers, attorney` の追加でnDCG@10が0から1へ改善しました。質問691では `payment, cat, cards` の追加で1から0へ悪化しました。追加語が必ずしも同義語ではないことと、元の語と同じ強さで扱う場合の検索意図のずれが考えられます。個別例から一般的な原因を断定はできません。

全問の追加語と評価値は `results/fiqa/prefixes.json` に、最初に実行した3語比較は `results/fiqa/comparison_3terms_min2.json` に保存しました。FiQAでも未判定文書は非関連として扱います。

### Recallのカットオフを広げると

`compare_prefixes.py` で上位100件まで取り、Recall@10/20/50/100を計算しました。`candidate_recall` は、上位件数を切らず、BM25で得点が付いた文書に正解が含まれる割合です。

| データ・条件 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | candidate_recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact・BM25 | 0.78761 | 0.82922 | 0.87039 | 0.88589 | 0.99333 |
| SciFact・3語 | 0.78578 | 0.81422 | 0.86022 | 0.89589 | 0.99333 |
| FiQA・BM25 | 0.29823 | 0.36081 | 0.44610 | 0.50861 | 0.97559 |
| FiQA・1語 | 0.29411 | 0.36147 | 0.46107 | 0.51843 | 0.98058 |
| FiQA・2語 | 0.29503 | 0.36236 | 0.45092 | 0.53298 | 0.98228 |
| FiQA・3語 | 0.28161 | 0.34704 | 0.45182 | 0.51112 | 0.98472 |

FiQAの3語条件では、Recall@10が下がった質問が76問あり、そのうち61問ではRecall@100がBM25のみと同じか上でした。質問691「How to categorize credit card payments?」では、正解文書はBM25のみで1位でしたが、`payment, cat, cards` を追加すると上位20件から落ち、上位50件には残っています。この例では正解が候補から消えたのではなく、順位が押し下げられました。

結果ファイルは `results/scifact_recall_cutoffs.json` と `results/fiqa/recall_cutoffs.json` です。元の質問語を保持して語をORで加えるため、得点が付く候補集合は狭まりません。しかし、追加語に一致する非関連文書も上がるので、固定した上位件数内のRecallは下がり得ます。FiQAでは候補全体のRecallは上がった一方、Recall@10は下がりました。
