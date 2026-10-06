"""Run a decoded openpilot model through the NPU, without RKNN or NumPy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from driver import Device, Task
from tensor_io import pack, unpack
from layout import apply as apply_layout


class Model:
    def __init__(self, directory, device):
        self.directory = Path(directory)
        self.program = json.loads((self.directory / "program.json").read_text())
        program = self.program
        if program["format"] not in ("rk3588-decoded-model-v1", "rk3588-decoded-model-v2"):
            raise ValueError("Unsupported model program format")
        self.device = device
        self.buffers = {}
        for descriptor in program["buffers"]:
            payload = (self.directory / descriptor["file"]).read_bytes()
            if len(payload) != descriptor["payload_size"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Corrupt model buffer {descriptor['id']}")
            buffer = device.allocate(descriptor["size"], descriptor["flags"])
            buffer.write(payload)
            self.buffers[descriptor["id"]] = buffer
        self.task_buffer = self.buffers[program["task_buffer"]["buffer"]]
        if program["task_buffer"]["offset"]:
            raise ValueError("Offset kernel task objects have not been verified")
        self.relocations = 0
        for sequential, task in enumerate(program["tasks"], program["submit"]["task_start"]):
            index = task.get("index", sequential)
            if len(task["commands"]) != task["regcfg_amount"] + 4:
                raise ValueError("Task fetch must include four PC trailer words")
            location = task["regcmd"]
            commands = self.buffers[location["buffer"]]
            encoded = b"".join(struct.pack("<Q", self.encode(command)) for command in task["commands"])
            commands.write(encoded, location["offset"])
            if task["regcfg_offset"]:
                raise ValueError("Only absolute register descriptor addressing is verified")
            descriptor = Task(regcmd_addr=commands.dma + location["offset"])
            for key in ["flags", "op_idx", "enable_mask", "int_mask", "int_clear", "regcfg_amount", "regcfg_offset"]:
                setattr(descriptor, key, task[key])
            self.task_buffer.write(bytes(descriptor), index * 40)
        if self.relocations != program["export"]["relocated_pointers"]:
            raise ValueError("Address relocation count differs from export")
        for buffer in self.buffers.values():
            buffer.sync(1)

    def address(self, location):
        buffer = self.buffers[location["buffer"]]
        offset = location["offset"]
        if not 0 <= offset < buffer.size:
            raise ValueError("DMA pointer is outside the allocated model buffer")
        return buffer.dma + offset

    def encode(self, command):
        if command["target"] == "NOP":
            return 0
        definition = self.program["registers"][command["register"]]
        value = command.get("unlabelled_bits", 0)
        for name, field in command["fields"].items():
            low, high = definition["fields"][name]
            if field < 0 or field >= 1 << (high - low + 1):
                raise ValueError(f"Invalid {command['register']}.{name}")
            value |= field << low
        if "pointer" in command:
            pointer = command["pointer"]
            dma = self.address(pointer)
            if definition["offset"] == 0x10:
                if dma & 15:
                    raise ValueError("PC command stream address must be 16-byte aligned")
                value = dma | pointer["low_bits"]
            else:
                value = dma
            self.relocations += 1
        target = self.program["targets"][command["target"]]
        return target << 48 | value << 16 | definition["offset"]

    def run_bytes(self, inputs, canonical=False):
        io = self.program["io"]
        expected = {item["attr"]["name"] for item in io if item["kind"] == "input"}
        if set(inputs) != expected:
            raise ValueError(f"Model input names must be {sorted(expected)}")
        for item in io:
            memory = item["memory"]
            buffer = self.buffers[memory["buffer"]]
            if item["kind"] == "input":
                raw = inputs[item["attr"]["name"]]
                if canonical:
                    raw = pack(raw, item.get("logical", item["attr"]), item["attr"])
                if len(raw) != item["attr"]["size_with_stride"]:
                    raise ValueError("Input byte count differs from native model tensor")
                buffer.write(raw, memory["offset"])
                buffer.sync(1)
            else:
                # Detect incomplete execution instead of returning an old output.
                size = item["attr"]["size_with_stride"]
                buffer.write(b"\x00\x7e" * (size // 2), memory["offset"])
                buffer.sync(1)
        for sequence, settings in enumerate(self.program.get("submits", [self.program["submit"]]), 1):
            for operation in self.program.get("layout_operations", []):
                if operation["before_submit"] == sequence:
                    apply_layout(operation, self.buffers)
            self.device.submit(self.task_buffer, settings)
        results = {}
        for item in io:
            if item["kind"] != "output":
                continue
            attr = item["attr"]
            if attr["type"] != 1:
                raise ValueError("This model output decoder requires FP16 tensors")
            memory = item["memory"]
            buffer = self.buffers[memory["buffer"]]
            buffer.sync(2)
            start = memory["offset"]
            native = buffer.mapping[start:start + attr["size_with_stride"]]
            raw = unpack(native, item.get("logical", attr), attr)
            values = [value for (value,) in struct.iter_unpack("<e", raw)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("NPU output contains unwritten/nonfinite values")
            results[attr["name"]] = raw
        return results

    def run(self, inputs, canonical=False):
        return {name: [value for (value,) in struct.iter_unpack("<e", raw)]
                for name, raw in self.run_bytes(inputs, canonical=canonical).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["navigation", "dmonitoring", "supercombo"])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--input", action="append", default=[], help="NAME=FILE with little-endian FP16 bytes; repeat for each input")
    parser.add_argument("--canonical", action="store_true", help="Input files use canonical tensor order")
    parser.add_argument("--case", choices=["fixture", "map", "zeros", "ramp"], default="fixture")
    parser.add_argument("--output", type=Path, help="Write little-endian FP32 output")
    parser.add_argument("--verify", action="store_true", help="Compare the packaged fixture with RKNN")
    parser.add_argument("--dry-run", action="store_true", help="Inspect program without opening the device")
    args = parser.parse_args()
    directory = args.cache or Path.home() / ".cache/rk3588-models/openpilot" / args.model
    program = json.loads((directory / "program.json").read_text())
    if args.dry_run:
        print(json.dumps({"model": program["model"], "submit": program["submit"], "export": program["export"]}, indent=2))
        return
    input_items = [item for item in program["io"] if item["kind"] == "input"]
    inputs = {}
    if args.input:
        for specification in args.input:
            if "=" in specification:
                name, filename = specification.split("=", 1)
            elif len(input_items) == 1:
                name, filename = input_items[0]["attr"]["name"], specification
            else:
                raise ValueError("Multiple-input models require NAME=FILE")
            if name in inputs:
                raise ValueError(f"Duplicate input {name}")
            inputs[name] = Path(filename).read_bytes()
    else:
        if args.canonical:
            raise ValueError("--canonical applies to explicit input files")
        for item in input_items:
            attr = item["attr"]
            name = attr["name"]
            if args.case in ("fixture", "map"):
                filename = directory / f"input-{name}.f16"
                inputs[name] = (filename if filename.exists() else directory / "input.f16").read_bytes()
            elif args.case == "zeros":
                inputs[name] = bytes(attr["size_with_stride"])
            else:
                if args.model != "navigation":
                    raise ValueError("The normalized ramp fixture is specific to navigation")
                inputs[name] = b"".join(struct.pack("<e", index / (256 * 256 - 1)) for index in range(256 * 256))
    if args.verify and (args.input or args.case not in ("fixture", "map")):
        raise ValueError("Packaged RKNN verification applies to the supplied fixture only")
    start = time.monotonic()
    with Device() as device:
        model = Model(directory, device)
        results = model.run(inputs, canonical=args.canonical)
        if len(results) != 1:
            raise ValueError("The example CLI expects one output tensor")
        output = next(iter(results.values()))
    if args.output:
        args.output.write_bytes(struct.pack(f"<{len(output)}f", *output))
    result = {"model": args.model, "backend": "registers", "elements": len(output),
              "finite": True, "min": min(output), "max": max(output),
              "elapsed_seconds": time.monotonic() - start,
              "relocated_pointers": model.relocations}
    if args.verify:
        reference = [value for (value,) in struct.iter_unpack("<f", (directory / "rknn-output.f32").read_bytes())]
        if len(reference) != len(output):
            raise ValueError("Reference output length mismatch")
        error = max(abs(a - b) for a, b in zip(output, reference))
        result["max_abs_vs_rknn"] = error
        if error != 0:
            raise RuntimeError(f"Register output differs from RKNN: max abs {error}")
        result["verified"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
