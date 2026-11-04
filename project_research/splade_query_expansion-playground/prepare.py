"""Download fixed BEIR archives and the SPLADE checkpoint for this experiment."""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATASETS = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "fiqa": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
    "arguana": "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
}
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
MODEL = "naver/splade-cocondenser-ensembledistil"
MODEL_REVISION = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="scifact")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    archive = DATA / f"{args.dataset}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(DATASET_URL.format(name=args.dataset), archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != DATASETS[args.dataset]:
        raise ValueError(f"{args.dataset} archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {name}")
        zipped.extractall(DATA)
    print(f"{args.dataset} ready", flush=True)
    snapshot_download(
        repo_id=MODEL,
        revision=MODEL_REVISION,
        local_dir=DATA / "model",
        allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"],
        max_workers=1,
    )
    (DATA / "model_revision.txt").write_text(MODEL_REVISION + "\n")
    print("SPLADE model ready")


if __name__ == "__main__":
    main()
