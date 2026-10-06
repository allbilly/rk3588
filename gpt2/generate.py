"""GPT-2 124M autoregressive inference using decoded RK3588 NPU registers.

All embedding addition, normalization, attention, MLPs and the vocabulary
projection execute on the NPU. Python handles tokenization, byte layout copies,
KV cache bookkeeping and greedy token selection. No third-party imports.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpilot"))
from infer import Device, Model
from checkpoint import Checkpoint
from tokenizer import Tokenizer


class Decoder:
    context = 1024

    def __init__(self, directory, device):
        self.directory = Path(directory)
        self.checkpoint = Checkpoint(self.directory / "checkpoint/model.safetensors")
        self.embedding = Model(self.directory / "embedding", device)
        self.head = Model(self.directory / "head", device)
        self.layers = [(Model(self.directory / f"layer-{index:02d}-qkv", device),
                        Model(self.directory / f"layer-{index:02d}-attention-ffn", device))
                       for index in range(12)]
        self.reset()

    def reset(self):
        self.position = 0
        # Canonical K: [head, dimension, time]; V: [head, time, dimension].
        self.keys = [bytearray(12 * 64 * self.context * 2) for _ in self.layers]
        self.values = [bytearray(12 * self.context * 64 * 2) for _ in self.layers]

    def step(self, token, logits=True):
        position = self.position
        if position >= self.context:
            raise ValueError("GPT-2 context is limited to 1024 tokens")
        hidden = self.embedding.run_bytes({"token": self.checkpoint.row("wte.weight", token),
                                           "position": self.checkpoint.row("wpe.weight", position)},
                                          canonical=True)["output"]
        mask = (bytes((position + 1) * 2) + struct.pack("<e", -10000) *
                (self.context - position - 1)) * 12
        for index, (projection, attention) in enumerate(self.layers):
            qkv = projection.run_bytes({"hidden": hidden}, canonical=True)["output"]
            if len(qkv) != 2304 * 2:
                raise ValueError("Unexpected QKV projection size")
            key, value = self.keys[index], self.values[index]
            for head in range(12):
                for dimension in range(64):
                    source = (768 + head * 64 + dimension) * 2
                    target = ((head * 64 + dimension) * self.context + position) * 2
                    key[target:target + 2] = qkv[source:source + 2]
                source = (1536 + head * 64) * 2
                target = (head * self.context + position) * 64 * 2
                value[target:target + 128] = qkv[source:source + 128]
            hidden = attention.run_bytes({"query": qkv[:1536], "keys": key,
                                          "values": value, "mask": mask, "residual": hidden},
                                         canonical=True)["output"]
        self.position += 1
        if logits:
            return self.head.run({"hidden": hidden}, canonical=True)["output"]
        return None

    def close(self):
        self.checkpoint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=10, help="Number of greedy continuation tokens")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/rk3588-models/gpt2")
    parser.add_argument("--logits", type=Path, help="Save the prompt's next-token logits as little-endian FP32")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tokenizer = Tokenizer(args.cache / "checkpoint")
    prompt = tokenizer.encode(args.prompt) or [tokenizer.eos]
    if args.tokens < 0 or len(prompt) + args.tokens > Decoder.context:
        raise ValueError("Prompt plus continuation must fit 1024 tokens")
    start = time.monotonic()
    generated = []
    with Device() as device:
        decoder = Decoder(args.cache, device)
        try:
            for index, token in enumerate(prompt):
                scores = decoder.step(token, logits=index == len(prompt) - 1)
            if args.logits:
                args.logits.write_bytes(struct.pack(f"<{len(scores)}f", *scores))
            for index in range(args.tokens):
                token = max(range(len(scores)), key=scores.__getitem__)
                generated.append(token)
                if token == tokenizer.eos or index == args.tokens - 1:
                    break
                scores = decoder.step(token)
        finally:
            decoder.close()
    result = {"model": "gpt2-124m", "backend": "registers", "prompt_tokens": prompt,
              "generated_tokens": generated, "text": tokenizer.decode(prompt + generated),
              "elapsed_seconds": time.monotonic() - start}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["text"])


if __name__ == "__main__":
    main()
