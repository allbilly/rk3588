"""Fetch the pinned GPT-2 124M checkpoint using only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FILES = {
    "config.json": (665, "git", "10c66461e4c109db5a2196bff4bb59be30396ed8"),
    "merges.txt": (456318, "git", "226b0752cac7789c48f0cb3ec53eda48b7be36cc"),
    "vocab.json": (1042301, "git", "1f1d9aaca301414e7f6c9396df506798ff4eb9a6"),
    "model.safetensors": (548105171, "sha256", "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"),
}


def verify(path, size, kind, expected):
    if not path.exists() or path.stat().st_size != size:
        return False
    digest = hashlib.sha1() if kind == "git" else hashlib.sha256()
    if kind == "git":
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/rk3588-models/gpt2/checkpoint")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for name, (size, kind, digest) in FILES.items():
        target = args.cache / name
        if verify(target, size, kind, digest):
            print(f"Verified cached {name}", flush=True)
            continue
        temporary = args.cache / (name + ".part")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true"
        request = urllib.request.Request(url, headers={"User-Agent": "rk3588-python-model-example", "Cache-Control": "no-cache"})
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(4 * 1024 * 1024):
                file.write(chunk)
                total += len(chunk)
                if total % (64 * 1024 * 1024) == 0:
                    print(f"{name}: {total // (1024 * 1024)} / {size // (1024 * 1024)} MiB", flush=True)
        if not verify(temporary, size, kind, digest):
            raise RuntimeError(f"Downloaded {name} failed its pinned size/hash check")
        temporary.replace(target)
        print(f"Verified {name}", flush=True)
    config = json.loads((args.cache / "config.json").read_text())
    if (config["n_embd"], config["n_layer"], config["n_head"], config["vocab_size"], config["n_positions"]) != (768, 12, 12, 50257, 1024):
        raise ValueError("Checkpoint is not GPT-2 124M")
    print(f"Checkpoint ready: {args.cache}", flush=True)


if __name__ == "__main__":
    main()
