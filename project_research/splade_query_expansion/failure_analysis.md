# 追加語で順位が悪化した例

条件は[README](README.md)の0語対3語。下の「元語」「追加語」は各文書のBM25への寄与であり、SPLADEのスコアではない。数値は小数第2位に丸めた。正解はBEIRのtest qrelsで判定している。

| データ・質問 | 正解の変化 | 追加語 | 観察 |
| --- | --- | --- | --- |
| FiQA 691「How to categorize credit card payments?」 | 1位→20位圏外、50位以内 | `payment, cat, cards` | 正解文書の元語18.54＋追加語0。新1位のカード一般論は元語15.09＋`payment` 3.52＋`cards` 6.18で追い抜く。目立つ`cat`はこの新1位には出現しない。 |
| FiQA 9737「Long(100%)-Short(-100%) investment explanation」 | 正解3文書が2・3・4位→全て10位圏外、20位以内 | `extended, percent, investments` | 具体的なポジション比率の説明より、投資一般の文書が上がる。新1位は元語11.51＋`percent` 6.25、確認した正解文書は元語13.56＋追加語0。 |
| FiQA 7467「Stocks are traded on secondary markets?」 | 1位→6位 | `stock, primary, market` | `secondary`を尋ねる質問に`primary`も追加。新1位は一次・二次市場を比較する文書で、元語19.41＋追加語11.30。正解文書は元語20.43＋追加語3.06。新1位も内容は関連しており、qrels未判定の影響があり得る。 |
| SciFact 660「Ivermectin is used to treat onchocerciasis.」 | 1位→20位圏外、50位以内 | `iv, drug, treatment` | 正解文書は元語8.81＋追加語0。新1位はイブプロフェン治療の論文で、元語3.35＋`iv` 7.36＋`drug` 1.44＋`treatment` 2.32。`iv`は検索意図に合う同義語として扱えない。 |
| SciFact 1303「Tirasemtiv has no effect on fast-twitch muscle.」 | 1位→7位 | `ti, muscles, yes` | 新1位の別の筋肉研究は元語10.16＋`muscles` 10.05。正解文書は元語15.29＋追加語0。`yes`は気になるが、この新1位を押し上げた直接の語は`muscles`。 |

## 見えている性質

1. **追加語を元の語と同じ強さで足すと、関連の薄い文書が逆転する。** FiQA 691では正解文書が追加語を一つも含まず、カード一般論が`payment`と`cards`の加点で追い抜いた。SciFact 660も同じ構図。これはモデルの語の重みを捨て、BM25のIDFで再重み付けしている実験条件の影響。
2. **候補語は同義語に限らない。** `iv`のような曖昧な短語、`percent`や`investments`のような広い語、`secondary`に対する`primary`のような関連概念も混ざる。語が文脈上関連していても、無重みのOR検索で加点する語として有効とは限らない。
3. **短語だけが全体悪化の原因ではない。** FiQAでは追加3語に3文字以下が含まれる質問が235/648問、nDCG@10悪化質問では60/164問。短語あり・なしの悪化率はそれぞれ約25.5%・25.2%でほぼ同じ。SciFactで4文字以上の3語に限定した探索的条件もnDCG@10が0.65337から0.64233へ下がった。
4. **評価ラベルには限界がある。** FiQA 7467の新1位は一次・二次市場を説明しており、質問に全く無関係とは言い切れない。未判定文書を非関連として計算しているため、各例の「悪化」はこのqrels上の評価を指す。

## モデルと手法の切り分け

今回の [`naver/splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil) は2022年のSPLADE++系の公開モデルで、モデルカードはMS MARCO devのMRR@10を38.3と報告している。少なくとも極端に弱いモデルを選んだとは言いにくい。ただしMS MARCO系で学習され、科学・金融の語彙や質問に合わない可能性はある。特に固有名詞・専門語や短い略語は注意が必要。

より大きな問題は**モデルの出力の使い方が本来のSPLADE検索と違う**こと。SPLADEはクエリと文書をともに疎ベクトルへ変換し、語ごとの学習済み重みの内積で順位付けする。この実験はクエリ側の上位語だけを取り、元の文書にBM25を適用する。したがって今回の悪化は「SPLADE自体の検索性能が低い」とは示さない。新しい[SPLADE-v3](https://arxiv.org/abs/2403.06789)への変更だけで解消するとも断定できない。モデル差と統合方法を区別するなら、同じ追加語で追加分のBM25寄与を弱める条件を先に比較するのが小さい次実験になる。

個々の順位と全指標は `results/scifact_recall_cutoffs.json`、`results/fiqa/recall_cutoffs.json` に保存した。
