"""Verify all openpilot fixtures and GPT-2 reference prompts on the NPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import struct
import sys
import time

from infer import Device, Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt2"))
from generate import Decoder, Tokenizer


def verify_openpilot(directory, device):
    model = Model(directory, device)
    inputs = {}
    for item in model.program["io"]:
        if item["kind"] == "input":
            name = item["attr"]["name"]
            path = directory / f"input-{name}.f16"
            inputs[name] = (path if path.exists() else directory / "input.f16").read_bytes()
    reference = [value for value, in struct.iter_unpack("<f", (directory / "rknn-output.f32").read_bytes())]
    start = time.monotonic()
    for repeat in range(2):
        result = model.run(inputs)
        if len(result) != 1:
            raise ValueError("Expected one openpilot output")
        values = next(iter(result.values()))
        if len(values) != len(reference) or values != reference:
            raise RuntimeError(f"{directory.name} differs from RKNN on repeat {repeat + 1}")
    return {"model": directory.name, "elements": len(reference), "repeats": 2,
            "max_abs_vs_rknn": 0.0, "seconds": time.monotonic() - start}


def verify_gpt2(directory, device):
    reference = json.loads((Path(__file__).resolve().parents[1] / "gpt2/validation.json").read_text())
    tokenizer = Tokenizer(directory / "checkpoint")
    decoder = Decoder(directory, device)
    results = []
    try:
        for case in reference["cases"]:
            decoder.reset()
            start = time.monotonic()
            prompt = tokenizer.encode(case["prompt"])
            for index, token in enumerate(prompt):
                scores = decoder.step(token, logits=index == len(prompt) - 1)
            generated = []
            expected = case["reference_generated_tokens"]
            for index in range(len(expected)):
                token = max(range(len(scores)), key=scores.__getitem__)
                generated.append(token)
                if token == tokenizer.eos or index == len(expected) - 1:
                    break
                scores = decoder.step(token)
            if generated != expected:
                raise RuntimeError(f"GPT-2 {case['prompt']!r}: expected {expected}, got {generated}")
            result = {"prompt": case["prompt"], "generated_tokens": generated,
                      "tokens_match": True, "seconds": time.monotonic() - start}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        decoder.close()
    return {"model": "gpt2-124m", "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=["auto", "rknpu", "rocket"], default="auto")
    parser.add_argument("--device", help="NPU node; checked against its sysfs driver")
    parser.add_argument("--openpilot-cache", type=Path,
                        default=Path.home() / ".cache/rk3588-models/openpilot")
    parser.add_argument("--gpt2-cache", type=Path, default=Path.home() / ".cache/rk3588-models/gpt2")
    parser.add_argument("--report", type=Path, help="Save the complete successful verification report")
    args = parser.parse_args()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "kernel": platform.release(), "machine": platform.machine(), "models": []}
    for name in ("navigation", "dmonitoring", "supercombo", "gpt2"):
        with Device(args.device, driver=args.driver) as device:
            report["driver"], report["device"] = device.driver, device.path
            result = (verify_gpt2(args.gpt2_cache, device) if name == "gpt2" else
                      verify_openpilot(args.openpilot_cache / name, device))
        report["models"].append(result)
        print(json.dumps({"model": name, "passed": True, "driver": device.driver}), flush=True)
    report["passed"] = True
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
