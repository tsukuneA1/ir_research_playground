"""固定したSPLADE-v3のモデルとトークナイザをCPUに読み込む。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "naver/splade-v3"
MODEL_REVISION = "fdfeceb91d7b9de7985b38addd3ba9f53a59a355"
# モデルカードの評価再現設定。後続の推論でもtruncation=Trueと共に指定する。
MAX_INPUT_TOKENS = 256
CACHE_DIR = Path(__file__).resolve().parent / "data" / "models"


@dataclass(frozen=True)
class LoadedSplade:
    """同一コミットから読み込んだモデルとトークナイザ、および入力長の設定。"""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    max_input_tokens: int = MAX_INPUT_TOKENS


def load_splade(*, local_files_only: bool = False) -> LoadedSplade:
    """SPLADE-v3を固定コミットから読み込み、CPU・評価モードに設定する。

    トークナイザとMasked LMの重みを同じコミットから取得する。CPUの
    スレッド数は2、重みの精度はfloat32、入力長は特殊トークン込み256とする。
    トークナイザのmodel_max_lengthは後続処理への設定値であり、それだけでは
    入力は切り詰められない。推論時はtruncation=Trueを明示する必要がある。

    初回の取得先は実験ディレクトリ内のdata/models。取得にはHugging Faceの
    モデルページでの条件同意と、権限を持つアカウントでの認証が必要になる。
    取得・認証のエラーは呼び出し元へ伝える。

    引数:
        local_files_only: Trueなら取得済みキャッシュだけを使用し、通信しない。

    戻り値:
        CPU上の評価モードのモデルとトークナイザ。推論や疎ベクトルの
        pooling、拡張語の選択はこの関数では行わない。
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        model_max_length=MAX_INPUT_TOKENS,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    model.to("cpu")
    model.eval()
    return LoadedSplade(model, tokenizer)
