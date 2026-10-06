"""Convert a read-only RKNN capture into a relocatable decoded NPU program.

No RKNN library is used here. The source capture and register XML are inputs to
this setup step; the inference runtime consumes only the generated cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from tensor_io import unpack
from layout import transpose
from tensor_io import pack

TARGETS = {0: "NOP", 0x41: "VERSION", 0x81: "PC_ENABLE", 0x101: "PC_CONFIG",
           0x201: "CNA", 0x801: "CORE", 0x1001: "DPU", 0x2001: "DPU_RDMA",
           0x4001: "PPU", 0x8001: "PPU_RDMA"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=Path.home() / "npu/ops_rknn/registers.xml")
    parser.add_argument("--layout", type=Path, help="Explicit byte-copy operators between submissions")
    args = parser.parse_args()
    if (args.output / "program.json").exists():
        raise ValueError("Output already contains a program; use a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.capture
    reference = json.loads((source / "reference.json").read_text())
    allocations = [json.loads(line) for line in (source / "before-0001-memory.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (source / "submissions.jsonl").read_text().splitlines()]
    submits = [row for row in rows if "task" not in row]
    operations = json.loads(args.layout.read_text()) if args.layout else []
    for submit in submits:
        if not submit["mapped"] or submit["core_mask"] != 1 or submit["flags"] not in (1, 5):
            raise ValueError("Expected captured, blocking, single-core PC submissions")
        active_start, active_count = submit["subcores"][0]
        active_indices = set(range(active_start, active_start + active_count))
        if any(not row["mapped"] for row in rows
               if row.get("submit") == submit["submit"] and row.get("task") in active_indices):
            raise ValueError("Capture contains unmapped active tasks or register buffers")
    # Never bake a captured intermediate into a supposedly dynamic model.
    for sequence in range(1, len(submits)):
        snapshot = {a["id"]: bytearray((source / f"after-{sequence:04d}-bo-{a['id']:04d}.bin").read_bytes())
                    for a in allocations}
        for operation in operations:
            if operation["before_submit"] != sequence + 1:
                continue
            origin, target = operation["source"], operation["destination"]
            start = origin["memory"]["offset"]
            raw = snapshot[origin["memory"]["buffer"]][start:start + origin["native"]["size_with_stride"]]
            moved = transpose(unpack(raw, origin["logical"], origin["native"]),
                              origin["logical"]["dims"], operation["permutation"])
            packed = pack(moved, target["logical"], target["native"])
            start = target["memory"]["offset"]
            snapshot[target["memory"]["buffer"]][start:start + len(packed)] = packed
        for allocation in allocations:
            old = source / f"after-{sequence:04d}-bo-{allocation['id']:04d}.bin"
            new = source / f"before-{sequence + 1:04d}-bo-{allocation['id']:04d}.bin"
            if bytes(snapshot[allocation["id"]]) != new.read_bytes():
                raise ValueError(f"CPU memory changes before submit {sequence + 1}, buffer {allocation['id']}; explicit layout operator required")

    def locate(address, length=1, object_address=False):
        key = "obj" if object_address else "dma"
        matches = [a for a in allocations
                   if (a[key] == address and length <= a["size"] if object_address
                       else a[key] <= address and address + length <= a[key] + a["size"])]
        if len(matches) != 1:
            raise ValueError(f"Unresolved/ambiguous DMA range {address:#x} + {length}")
        return {"buffer": matches[0]["id"], "offset": address - matches[0][key]}

    ns = {"r": "http://nouveau.freedesktop.org/"}
    xml = ET.parse(args.xml).getroot()
    definitions = {}
    for domain in xml.findall("r:domain", ns):
        for register in domain.findall("r:reg32", ns):
            offset = int(register.attrib["offset"], 0)
            fields = {}
            for field in register.findall("r:bitfield", ns):
                low = int(field.attrib.get("low", field.attrib.get("pos", "0")))
                high = int(field.attrib.get("high", str(low)))
                fields[field.attrib["name"]] = [low, high]
            definitions[offset] = {"name": domain.attrib["name"] + "." + register.attrib["name"],
                                   "offset": offset, "fields": fields}
    # These zero-valued reserved registers are observed in existing conv examples.
    for offset, name in [(0x3030, "CORE.RESERVED_3030"), (0x40C4, "DPU.RESERVED_40C4")]:
        definitions[offset] = {"name": name, "offset": offset, "fields": {"RESERVED": [0, 31]}}
    used = {}
    buffers = {a["id"]: bytearray((source / a["file"]).read_bytes()) for a in allocations}
    tasks = []
    pointers = 0
    unknown_bits = 0
    task_buffer = None
    for submit in submits:
        active_start, active_count = submit["subcores"][0]
        active_indices = set(range(active_start, active_start + active_count))
        task_data = (source / f"submit-{submit['submit']:04d}-tasks.bin").read_bytes()
        if len(task_data) % 40 or len(task_data) < active_count * 40:
            raise ValueError("Task descriptor length does not match submit")
        current_task_buffer = locate(submit["task_obj_addr"], (active_start + active_count) * 40, True)
        if task_buffer is None:
            task_buffer = current_task_buffer
        elif task_buffer != current_task_buffer:
            raise ValueError("Multiple task objects require separate programs")
        for index, values in enumerate(struct.iter_unpack("<8IQ", task_data), submit["task_start"]):
            if index not in active_indices:
                continue
            flags, op_idx, enable, int_mask, int_clear, status, amount, reg_offset, address = values
            raw = (source / f"submit-{submit['submit']:04d}-task-{index:05d}-fetch.bin").read_bytes()
            if len(raw) != (amount + 4) * 8 or reg_offset:
                raise ValueError("Unexpected PC fetch length or descriptor offset mode")
            location = locate(address, len(raw))
            start = location["offset"]
            buffers[location["buffer"]][start:start + len(raw)] = bytes(len(raw))
            commands = []
            for (word,) in struct.iter_unpack("<Q", raw):
                target, offset, value = word >> 48, word & 0xffff, (word >> 16) & 0xffffffff
                if target not in TARGETS:
                    raise ValueError(f"Unknown command target {target:#x}")
                if target == 0:
                    if word:
                        raise ValueError("Nonzero NOP command")
                    commands.append({"target": "NOP"})
                    continue
                definition = definitions[offset]
                name = definition["name"]
                if name.endswith(("RESERVED_3030", "RESERVED_40C4")) and value:
                    raise ValueError("Reserved register contains an unexplained nonzero value")
                used[name] = definition
                mask = 0
                fields = {}
                for field, (low, high) in definition["fields"].items():
                    bits = ((1 << (high - low + 1)) - 1) << low
                    mask |= bits
                    fields[field] = (value & bits) >> low
                command = {"target": TARGETS[target], "register": name, "fields": fields}
                if value & ~mask:
                    command["unlabelled_bits"] = value & ~mask
                    unknown_bits += 1
                pointer = any(part in name for part in ("BASE_ADDR", "FEATURE_DATA_ADDR", "DCOMP_ADDR"))
                if pointer and value:
                    dma = value & ~15 if offset == 0x10 else value
                    command["pointer"] = locate(dma)
                    command["pointer"]["low_bits"] = value & 15 if offset == 0x10 else 0
                    # Captured absolute addresses never become runtime constants.
                    for field, (low, high) in definition["fields"].items():
                        if (offset == 0x10 and low == 4) or (low == 0 and high == 31):
                            command["fields"][field] = 0
                    pointers += 1
                commands.append(command)
            tasks.append({"index": index, "flags": flags, "op_idx": op_idx, "enable_mask": enable,
                          "int_mask": int_mask, "int_clear": int_clear,
                          "regcfg_amount": amount, "regcfg_offset": reg_offset,
                          "regcmd": location, "commands": commands})

    # Rebuild task descriptors in Python; no captured object/DMA addresses remain.
    buffers[task_buffer["buffer"]] = bytearray(len(buffers[task_buffer["buffer"]]))
    io = []
    for item in reference["io"]:
        memory = locate(item["dma"], item["size"])
        if item["kind"] == "output":
            # Old RKNN models copy internal outputs to the public zero-copy BO.
            # Use the final active task's actual destination, and prove that its
            # post-submit bytes agree with the public native output.
            destinations = [c["pointer"] for task in tasks for c in task["commands"]
                            if c.get("register") == "DPU.DST_BASE_ADDR" and "pointer" in c]
            size = item["attr"]["size_with_stride"]
            logical = item.get("logical", item["attr"])
            expected = [value for (value,) in struct.iter_unpack("<f", (source / "rknn-output.f32").read_bytes())]
            expected_raw = struct.pack(f"<{len(expected)}e", *expected)
            if [value for (value,) in struct.iter_unpack("<e", expected_raw)] != expected:
                raise ValueError("Reference is not exactly representable as native FP16")
            matches = []
            seen = set()
            final_memory = {}
            for destination in destinations:
                identity = (destination["buffer"], destination["offset"])
                if identity in seen:
                    continue
                seen.add(identity)
                allocation = next(a for a in allocations if a["id"] == destination["buffer"])
                if allocation["id"] not in final_memory:
                    final_memory[allocation["id"]] = (source / f"after-{len(submits):04d}-bo-{allocation['id']:04d}.bin").read_bytes()
                after = final_memory[allocation["id"]]
                candidate = after[destination["offset"]:destination["offset"] + size]
                if len(candidate) != size:
                    continue
                canonical = unpack(candidate, logical, item["attr"])
                location = {key: destination[key] for key in ["buffer", "offset"]}
                if canonical == expected_raw and location not in matches:
                    matches.append(location)
            if len(matches) != 1:
                raise ValueError(f"Expected one verified NPU output destination, found {len(matches)}")
            memory = matches[0]

        io.append({"kind": item["kind"], "attr": item["attr"],
                   "logical": item.get("logical", item["attr"]), "memory": memory})
    descriptors = []
    for allocation in allocations:
        number = allocation["id"]
        payload = bytes(buffers[number])
        filename = f"buffer-{number:04d}.bin"
        (args.output / filename).write_bytes(payload)
        descriptors.append({"id": number, "size": allocation["size"], "flags": allocation["flags"],
                            "file": filename, "sha256": digest(payload), "payload_size": len(payload)})
    program = {"format": "rk3588-decoded-model-v2", "model": reference["model"],
               "source_rknn_sha256": reference["rknn_sha256"],
               "targets": {name: number for number, name in TARGETS.items()},
               "registers": used, "buffers": descriptors, "io": io,
               "submit": {key: submits[0][key] for key in ["flags", "timeout", "task_start", "task_number", "core_mask", "subcores"]},
               "submits": [{key: job[key] for key in ["flags", "timeout", "task_start", "task_number", "core_mask", "subcores"]} for job in submits],
               "layout_operations": operations,
               "task_buffer": task_buffer, "tasks": tasks,
               "export": {"relocated_pointers": pointers, "unlabelled_register_bits": unknown_bits,
                          "configured_words": sum(t["regcfg_amount"] for t in tasks),
                          "active_descriptors": len(tasks)}}
    for filename in [p.name for p in source.glob("input*.f16")] + ["rknn-output.f32"]:
        (args.output / filename).write_bytes((source / filename).read_bytes())
    (args.output / "program.json").write_text(json.dumps(program, separators=(",", ":")) + "\n")
    print(json.dumps(program["export"]))


if __name__ == "__main__":
    main()
