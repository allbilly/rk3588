"""Verified downloads and atomic model-cache preparation using the stdlib."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import urllib.request


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source, destination):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size == source["size"] and file_digest(destination) == source["sha256"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False) as file:
        temporary = Path(file.name)
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "rk3588-model-preparation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(4 * 1024 * 1024):
                    file.write(chunk)
            file.close()
            if temporary.stat().st_size != source["size"] or file_digest(temporary) != source["sha256"]:
                raise ValueError(f"Public model failed its pinned size/SHA256 check: {source['url']}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def expand(template):
    if template["format"] != "rk3588-preparation-template-v1":
        raise ValueError("Unsupported preparation template")
    program = copy.deepcopy(template["program"])
    definitions = template["command_definitions"]
    for task in program["tasks"]:
        task["commands"] = [copy.deepcopy(definitions[index]) for index in task["commands"]]
    return program


def put(buffer, offset, raw):
    if offset < 0 or offset + len(raw) > len(buffer):
        raise ValueError("Preparation write exceeds the model buffer")
    buffer[offset:offset + len(raw)] = raw


def constants(buffer, writes):
    for write in writes:
        kind = write["type"]
        if kind not in ("u32", "u8", "f16", "f32"):
            raise ValueError("Unsupported constant representation")
        formats = {"u32": "I", "u8": "B", "f16": "e", "f32": "f"}
        raw = struct.pack(f"<{len(write['values'])}{formats[kind]}", *write["values"])
        put(buffer, write["offset"], raw)


def existing(directory, program):
    path = Path(directory) / "program.json"
    if not path.exists():
        return False
    actual = json.loads(path.read_text())
    comparison = copy.deepcopy(actual)
    # Older caches initialized input BOs with their captured fixture. Inputs are
    # overwritten before every run, so this is equivalent to zero initialization.
    input_buffers = {item["memory"]["buffer"] for item in program["io"] if item["kind"] == "input"}
    expected_buffers = {item["id"]: item for item in program["buffers"]}
    for item in comparison["buffers"]:
        if item["id"] in input_buffers:
            item["sha256"] = expected_buffers[item["id"]]["sha256"]
    if comparison != program:
        raise ValueError(f"Existing cache differs from the shipped template; choose a new --cache directory: {directory}")
    for item in actual["buffers"]:
        payload = Path(directory) / item["file"]
        if payload.stat().st_size != item["payload_size"] or file_digest(payload) != item["sha256"]:
            raise ValueError(f"Existing model buffer is corrupt: {payload}")
    print("VERIFIED", Path(directory).name, flush=True)
    return True


def create(directory, program, build_buffer, fixture=None):
    directory = Path(directory)
    if existing(directory, program):
        return
    if directory.exists():
        raise ValueError(f"Incomplete cache directory exists; choose a new --cache directory: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=directory.name + "."))
    try:
        for descriptor in program["buffers"]:
            payload = build_buffer(descriptor)
            if len(payload) != descriptor["payload_size"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Prepared model buffer differs from the verified template: {descriptor['id']}")
            (temporary / descriptor["file"]).write_bytes(payload)
        if fixture:
            for item in program["io"]:
                if item["kind"] == "input":
                    name = item["attr"]["name"]
                    raw = bytearray(item["attr"]["size_with_stride"])
                    constants(raw, fixture["inputs"][name])
                    (temporary / f"input-{name}.f16").write_bytes(raw)
            values = fixture["output"]
            (temporary / "rknn-output.f32").write_bytes(struct.pack(f"<{len(values)}f", *values))
        (temporary / "program.json").write_text(json.dumps(program, separators=(",", ":")) + "\n")
        temporary.rename(directory)
        print("PREPARED", directory.name, flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
