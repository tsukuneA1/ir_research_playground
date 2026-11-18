"""SPLADE-v3のモデルとトークナイザの読み込みだけを確認する。"""

import json

from huggingface_hub.errors import GatedRepoError

from splade_model import MODEL_ID, MODEL_REVISION, load_splade


def main() -> None:
    """モデルを読み込み、コミット・入力長・CPU配置・評価モードを表示する。"""
    print(f"Loading {MODEL_ID} at {MODEL_REVISION}", flush=True)
    try:
        loaded = load_splade()
    except (GatedRepoError, OSError) as error:
        # TransformersはGatedRepoErrorをOSErrorに包む場合がある。
        if not isinstance(error, GatedRepoError) and "gated" not in str(error).lower():
            raise
        raise SystemExit(
            "モデルを取得できません。https://huggingface.co/naver/splade-v3 で"
            "利用条件に同意し、同じアカウントで `uv run hf auth login` を"
            "実行してから再試行してください。"
        ) from None
    print(
        json.dumps(
            {
                "model_id": loaded.model_id,
                "revision": loaded.revision,
                "max_input_tokens": loaded.max_input_tokens,
                "tokenizer_max_length": loaded.tokenizer.model_max_length,
                "model_type": loaded.model.config.model_type,
                "vocabulary_size": loaded.model.config.vocab_size,
                "device": str(loaded.model.device),
                "dtype": str(loaded.model.dtype),
                "training": loaded.model.training,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
