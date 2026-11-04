# 追加語の重みとzero-hit fallback

保存済みのSPLADE上位3語を使い、同じBM25索引で次を比較した。モデルの再推論や文書側の拡張はしていない。

`score(d) = BM25(元の質問, d) + α × BM25(追加3語, d)`

`α=0` はBM25のみ、`α=1` は最初の等重み実験と同じ。`α=0.1, 0.25, 0.5` は探索的な条件であり、testの結果を見てから選んだ「最適値」と解釈しない。

| データ | α | nDCG@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0 | 0.66046 | 0.78761 | 0.88589 |
| SciFact | 0.1 | 0.66811 | 0.79178 | 0.89256 |
| SciFact | 0.25 | 0.67183 | 0.79344 | 0.89189 |
| SciFact | 0.5 | 0.67182 | 0.79911 | 0.88922 |
| SciFact | 1 | 0.65337 | 0.78578 | 0.89589 |
| FiQA | 0 | 0.23585 | 0.29823 | 0.50861 |
| FiQA | 0.1 | 0.23885 | 0.29906 | 0.51676 |
| FiQA | 0.25 | 0.24005 | 0.30367 | 0.52728 |
| FiQA | 0.5 | 0.23994 | 0.30415 | 0.53574 |
| FiQA | 1 | 0.21704 | 0.28161 | 0.51112 |
| ArguAna | 0 | 0.30830 | 0.65505 | 0.90896 |
| ArguAna | 0.1 | 0.30779 | 0.65220 | 0.90896 |
| ArguAna | 0.25 | 0.30882 | 0.65434 | 0.90896 |
| ArguAna | 0.5 | 0.30869 | 0.65434 | 0.91038 |
| ArguAna | 1 | 0.30835 | 0.65434 | 0.91394 |
| NFCorpus | 0 | 0.30806 | 0.14915 | 0.23688 |
| NFCorpus | 0.1 | 0.31497 | 0.15119 | 0.25554 |
| NFCorpus | 0.25 | 0.31369 | 0.15027 | 0.25760 |
| NFCorpus | 0.5 | 0.31005 | 0.14960 | 0.25693 |
| NFCorpus | 1 | 0.26954 | 0.13233 | 0.25604 |

0.25倍のBM25のみとの差は、nDCG@10でSciFact `+0.01136`、FiQA `+0.00420`。質問単位で5,000回復元抽出した95%区間は、それぞれ約 `[+0.00186, +0.02153]`、`[-0.00197, +0.01037]`。FiQAの区間は0を含む。両データとも0.25倍ではRecall@10も微増したが、その差の95%区間はいずれも0を含む。

[失敗例](failure_analysis.md)にも効果が見える。SciFact 660では等重みで正解が10位圏外へ落ちるが、0.25倍なら2位。FiQA 691でも等重みでは10位圏外、0.25倍では2位。0.25倍で全ての悪化が消えるわけではない。

## 追加した2ドメイン

ArguAna（議論・1,406問）では0.25倍のnDCG@10差は `+0.00052`、質問単位3,000回の復元抽出による95%区間は約 `[-0.00097, +0.00203]`。Recall@10差は `-0.00071`。重みを変えても上位10件の改善はほぼ見えなかった。等重みのRecall@100は `0.90896→0.91394` と少し上がる。SPLADE入力は前の2ドメインと同じ256トークン上限で、ArguAnaの質問の513/1,406問（36.5%）が切り詰められる。この制約を含んだ比較である。

NFCorpus（医療・栄養・323問）では等重みでnDCG@10が `0.30806→0.26954` と大きく下がった一方、0.1倍では `0.31497`、Recall@100は `0.23688→0.25554`。0.1倍のnDCG@10差は `+0.00692`、同じ復元抽出による95%区間は約 `[+0.00230, +0.01157]`。ただし0.1倍は同じtestデータ上で探索した候補であり、未使用データでの再確認が必要。

4ドメインの結果を通じて、等重みの追加はNFCorpusとFiQAでnDCG@10を下げ、SciFactとArguAnaでも改善は確認できなかった。小さい重みはSciFact・FiQA・NFCorpusでnDCG@10を上げる傾向があったが、ArguAnaではほぼ変わらない。全ドメインに一律の改善を主張する結果ではない。

## zero-hitの二つの意味

- **運用可能な定義:** 元のBM25が得点>0の文書を1件も返さない。SciFact 300問、FiQA 648問、ArguAna 1,406問では**0問**。NFCorpusでは**25/323問**あり、その時だけ`α=1`を使うfallbackで関連文書が上位10件に入ったのは3問、上位100件に入ったのは8問。全問平均ではnDCG@10 `0.30806→0.30925`、Recall@10 `0.14915→0.15073`、Recall@100 `0.23688→0.24348`。このfallbackは関連度ラベルなしで発火条件を判定できる。
- **正解が上位10件にない:** SciFactで58問、FiQAで332問。ただし関連度ラベルを見ないと判定できないため、実際の検索時のfallback条件にはできない。評価上の上限として、該当質問だけ等重みの追加語に切り替えるとnDCG@10はSciFact `0.66046→0.67169`、FiQA `0.23585→0.25013`。これは正解を使って切替先を決めた診断値で、運用可能な改善値ではない。

全問の値、zero-hit判定、追加語は `results/scifact_weights.json`、`results/fiqa/weights.json`、`results/arguana/weights.json`、`results/nfcorpus/weights.json` に保存した。再計算は次の通り。

```bash
python3 weight_and_fallback.py --dataset scifact --expansions results/expansions.json --output results/scifact_weights.json
python3 weight_and_fallback.py --dataset fiqa --expansions results/fiqa/expansions_3terms_min2.json --output results/fiqa/weights.json
python3 weight_and_fallback.py --dataset arguana --expansions results/arguana/expansions_3terms_min2.json --output results/arguana/weights.json
python3 weight_and_fallback.py --dataset nfcorpus --expansions results/nfcorpus/expansions_3terms_min2.json --output results/nfcorpus/weights.json
```
