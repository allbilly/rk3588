"""Pack GPT-2 checkpoint weights into verified decoded kernel templates.

Only byte layout and floating representation conversion happen on the CPU.
There is no model arithmetic here and no third-party dependency. Templates ship
as decoded data derived from complete, verified layer-0 register captures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
from checkpoint import Checkpoint

# Offsets were proved by exact full-tensor matches in the layer-0 coefficient
# BO. They refer to the 1024-context decoded template, not physical addresses.
SLOTS = {
    "qkv": [
        ("ln_1.weight", 0, "F16"), ("ln_1.bias", 1536, "F32"),
        ("attn.c_attn.weight", 4608, "T16X32"),
        ("attn.c_attn.bias", 3543552, "F32"),
    ],
    "attention-ffn": [
        ("attn.c_proj.weight", 17664, "T16X32"),
        ("attn.c_proj.bias", 1197312, "F32"),
        ("ln_2.weight", 1200384, "F16"), ("ln_2.bias", 1201920, "F32"),
        ("mlp.c_fc.weight", 1204992, "T16X32"),
        ("mlp.c_fc.bias", 5923584, "F32"),
        ("mlp.c_proj.weight", 5935872, "T16X32"),
        ("mlp.c_proj.bias", 10654464, "F32"),
    ],
}


def representation(checkpoint, name, layout):
    metadata, raw = checkpoint.tensor(name)
    if metadata["dtype"] != "F32":
        raise ValueError("Pinned checkpoint tensor must be FP32")
    if layout == "F32":
        return raw
    # Chunk conversion also handles the 154 MB vocabulary tensor without
    # creating a tuple containing all 38 million Python floats at once.
    half = b"".join(struct.pack(f"<{len(chunk) // 4}e", *struct.unpack(f"<{len(chunk) // 4}f", chunk))
                    for chunk in (raw[start:start + 1048576] for start in range(0, len(raw), 1048576)))
    if layout == "F16":
        return half
    if layout == "T16X32_ROWS":
        outputs, inputs = metadata["shape"]
        if inputs % 32:
            raise ValueError("Vocabulary matrix inputs must fit 32-channel tiles")
        packed = bytearray(len(half))
        destination = 0
        # The final vocabulary tile contains one real output row, stored
        # compactly. The compiler does not pad it to sixteen coefficient rows.
        for outer in range(0, outputs, 16):
            for inner in range(0, inputs, 32):
                for output in range(outer, min(outer + 16, outputs)):
                    source = (output * inputs + inner) * 2
                    packed[destination:destination + 64] = half[source:source + 64]
                    destination += 64
        return bytes(packed)
    if layout != "T16X32":
        raise ValueError(f"Unsupported coefficient layout: {layout}")
    inputs, outputs = metadata["shape"]
    if inputs % 32 or outputs % 16:
        raise ValueError("Matrix dimensions must fit 16x32 coefficient tiles")
    # Checkpoint Conv1D matrices are [input, output]. CNA consumes output-16,
    # input-32 tiles, with input channels contiguous inside each output row.
    packed = bytearray(len(half))
    destination = 0
    for outer in range(0, outputs, 16):
        for inner in range(0, inputs, 32):
            for output in range(outer, outer + 16):
                source = (inner * outputs + output) * 2
                for byte in (0, 1):
                    packed[destination + byte:destination + 64:2] = half[source + byte:source + 32 * outputs * 2:outputs * 2]
                destination += 64
    return bytes(packed)


def prepare(directory):
    directory = Path(directory)
    with Checkpoint(directory / "checkpoint/model.safetensors") as checkpoint:
        # All four decoded templates and constant recipes ship with the source.
        # Only the public checkpoint is needed to build a fresh cache.
        from templates import prepare_templates
        prepare_templates(directory, checkpoint, representation)
        for kind, slots in SLOTS.items():
            source = directory / f"layer-00-{kind}"
            template = json.loads((source / "program.json").read_text())
            descriptor = next(item for item in template["buffers"] if item["id"] == 1)
            coefficients = (source / descriptor["file"]).read_bytes()
            if hashlib.sha256(coefficients).hexdigest() != descriptor["sha256"]:
                raise ValueError("Template coefficients failed their checksum")
            # Confirm the complete layer-0 checkpoint tensor at every slot
            # before using this geometry for any other layer.
            for name, offset, layout in slots:
                original = representation(checkpoint, "h.0." + name, layout)
                if coefficients[offset:offset + len(original)] != original:
                    raise ValueError(f"Template does not contain the expected {name}")
            for layer in range(1, 12):
                target = directory / f"layer-{layer:02d}-{kind}"
                exists = (target / "program.json").exists()
                target.mkdir(exist_ok=True)
                program = json.loads(json.dumps(template))
                packed = bytearray(coefficients)
                records = []
                for name, offset, layout in slots:
                    raw = representation(checkpoint, f"h.{layer}.{name}", layout)
                    packed[offset:offset + len(raw)] = raw
                    records.append({"tensor": f"h.{layer}.{name}", "offset": offset,
                                    "bytes": len(raw), "layout": layout})
                for item in program["buffers"]:
                    filename = target / item["file"]
                    if item["id"] == 1:
                        item["sha256"] = hashlib.sha256(packed).hexdigest()
                    if not exists:
                        if item["id"] == 1:
                            filename.write_bytes(packed)
                        else:
                            shutil.copyfile(source / item["file"], filename)
                program["model"] = f"gpt2-layer-{layer:02d}-{kind}"
                program["preparation"] = {"method": "checkpoint-weight-packing",
                                           "register_template": template["model"],
                                           "checkpoint_sha256": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
                                           "coefficients": records}
                if exists:
                    if json.loads((target / "program.json").read_text()) != program:
                        raise ValueError(f"Existing program differs from the verified template: {target}")
                    for item in program["buffers"]:
                        raw = (target / item["file"]).read_bytes()
                        if len(raw) != item["payload_size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                            raise ValueError(f"Existing model buffer is corrupt: {target / item['file']}")
                    print("VERIFIED", target.name, flush=True)
                else:
                    (target / "program.json").write_text(json.dumps(program, separators=(",", ":")) + "\n")
                    print("PREPARED", target.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/rk3588-models/gpt2")
    args = parser.parse_args()
    prepare(args.cache)


if __name__ == "__main__":
    main()
