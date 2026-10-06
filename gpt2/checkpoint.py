"""Read safetensors without NumPy, safetensors, torch, or transformers."""
import hashlib
import json
import mmap
from pathlib import Path
import struct


class Checkpoint:
    def __init__(self, path, verify=True):
        from setup import FILES
        self.path = Path(path)
        expected_size, _, expected_hash = FILES["model.safetensors"]
        if self.path.stat().st_size != expected_size:
            raise ValueError("Checkpoint size differs from the pinned GPT-2 model")
        if verify:
            digest = hashlib.sha256()
            with self.path.open("rb") as file:
                for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError("Checkpoint SHA256 differs from the pinned model")
        self.file = self.path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        length = struct.unpack_from("<Q", self.mapping)[0]
        if length > 1024 * 1024 or length + 8 > expected_size:
            raise ValueError("Invalid safetensors metadata length")
        self.metadata = json.loads(self.mapping[8:8 + length])
        self.start = 8 + length
        for name, item in self.metadata.items():
            if name == "__metadata__":
                continue
            first, last = item["data_offsets"]
            if first < 0 or last < first or self.start + last > expected_size:
                raise ValueError(f"Tensor {name} exceeds checkpoint bounds")

    def tensor(self, name):
        item = self.metadata[name]
        first, last = item["data_offsets"]
        return item, self.mapping[self.start + first:self.start + last]

    def row(self, name, index):
        item = self.metadata[name]
        if len(item["shape"]) != 2 or item["dtype"] not in ["F32", "F16"]:
            raise ValueError("Embedding lookup requires a rank-two floating tensor")
        rows, columns = item["shape"]
        if not 0 <= index < rows:
            raise ValueError("Embedding index outside the model vocabulary/context")
        size = 4 if item["dtype"] == "F32" else 2
        start = self.start + item["data_offsets"][0] + index * columns * size
        raw = self.mapping[start:start + columns * size]
        if item["dtype"] == "F32":
            # Representation conversion and copying are input preparation;
            # embedding addition and all transformer arithmetic run on the NPU.
            return struct.pack(f"<{columns}e", *struct.unpack(f"<{columns}f", raw))
        return raw

    def close(self):
        self.mapping.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
