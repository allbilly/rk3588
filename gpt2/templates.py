"""Build all four base kernels from shipped decoded templates and a checkpoint."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpilot"))
from cache import constants, create, existing, expand, put


def prepare_templates(directory, checkpoint, representation):
    for name in ("embedding", "layer-00-qkv", "layer-00-attention-ffn", "head"):
        template = json.loads((Path(__file__).parent / "templates" / f"{name}.json").read_text())
        program = expand(template)
        destination = Path(directory) / name
        if existing(destination, program):
            continue

        def build_buffer(descriptor):
            result = bytearray(descriptor["payload_size"])
            recipe = template["initializers"].get(str(descriptor["id"]), {})
            for tensor in recipe.get("tensors", []):
                put(result, tensor["offset"], representation(checkpoint, tensor["name"], tensor["layout"]))
            # Checkpoint-derived tensor bytes are independently verified; these
            # sparse constants preserve the compiler's normalization/GELU tables
            # and the exact coefficient payload checksum.
            constants(result, recipe.get("constants", []))
            return result
        create(destination, program, build_buffer)
