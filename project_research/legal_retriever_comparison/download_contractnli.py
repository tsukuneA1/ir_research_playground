"""Download the ContractNLI part of LegalBench-RAG without extra packages."""

import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = "awinml/legalbench-rag"
REVISION = "2b9c248bc8179ef0908fd6ba01d50b156facd48b"
BASE = f"https://huggingface.co/datasets/{REPO}"
ROOT = Path(__file__).parent / "data"


def get(url: str) -> bytes:
  with urlopen(url, timeout=30) as response:
    return response.read()


def main() -> None:
  tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/corpus/contractnli?recursive=true"
  files = [item["path"] for item in json.loads(get(tree_url)) if item["type"] == "file"]
  files.append("benchmarks/contractnli.json")
  sums = get(f"{BASE}/resolve/{REVISION}/SHA256SUMS").decode()
  expected = {
    line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sums.splitlines()
  }

  for number, path in enumerate(files, 1):
    if path not in expected or not (
      path.startswith("corpus/contractnli/") or path == "benchmarks/contractnli.json"
    ):
      raise ValueError(f"Unexpected dataset path: {path}")
    target = ROOT / path
    if (
      target.is_file()
      and hashlib.sha256(target.read_bytes()).hexdigest() == expected[path]
    ):
      continue
    content = get(f"{BASE}/resolve/{REVISION}/{quote(path)}")
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected[path]:
      raise ValueError(f"Checksum mismatch: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    if number % 20 == 0 or number == len(files):
      print(f"Downloaded or checked {number}/{len(files)} files", flush=True)
  print(f"Ready: {len(files) - 1} contracts and 1 benchmark at {ROOT}")


if __name__ == "__main__":
  main()
