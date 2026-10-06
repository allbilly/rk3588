"""Prepare offline openpilot models from pinned public RKNN coefficient files.

The files supply coefficient bytes only. Decoded register definitions and task
schedules ship with this example. No RKNN compiler or runtime is imported.
"""
import argparse
import json
from pathlib import Path

from cache import constants, create, download, existing, expand, put


def prepare(name, directory):
    template = json.loads((Path(__file__).parent / "templates" / f"{name}.json").read_text())
    program = expand(template)
    destination = Path(directory) / name
    if existing(destination, program):
        return
    source = download(template["source"], Path(directory) / "sources" / template["source"]["file"])
    with source.open("rb") as file:
        def build_buffer(descriptor):
            result = bytearray(descriptor["payload_size"])
            recipe = template["initializers"].get(str(descriptor["id"]), {})
            for target, offset, length in recipe.get("copies", []):
                if offset < 0 or offset + length > template["source"]["size"]:
                    raise ValueError("Coefficient copy exceeds the public source file")
                file.seek(offset)
                raw = file.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete public coefficient source")
                put(result, target, raw)
            constants(result, recipe.get("constants", []))
            return result
        create(destination, program, build_buffer, template["fixture"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["all", "navigation", "dmonitoring", "supercombo"], nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/rk3588-models/openpilot")
    args = parser.parse_args()
    for name in (["navigation", "dmonitoring", "supercombo"] if args.model == "all" else [args.model]):
        prepare(name, args.cache)


if __name__ == "__main__":
    main()
