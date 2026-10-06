# GPT-2 124M using RK3588 NPU registers

The full 12-layer GPT-2 model generates text with Python's standard library.
Embedding addition, layer normalization, QKV projections, scaled causal
attention, GELU, feed-forward projections, residual additions, final
normalization and all 50,257 vocabulary logits execute on the NPU.

Python performs byte BPE tokenization, embedding lookup, FP32-to-FP16
representation conversion, tensor layout copies, KV cache bookkeeping and
greedy token selection. There is no CPU or GPU neural network fallback.

## Run on this board

```sh
cd ~/rk3588
python3 -S gpt2/generate.py --prompt 'The capital of France is' --tokens 5
python3 -S gpt2/generate.py --prompt 'Hello' --tokens 5 --json
```

The first command produced:

```text
The capital of France is the capital of the French
```

This is the original pretrained GPT-2, not an instruction-tuned chat model.
`--tokens` controls the number of greedy continuation tokens. Prompt plus
continuation must fit the 1024-token context. `--logits FILE` saves the prompt's
next-token logits as little-endian FP32 values. `--cache DIRECTORY` selects a
prepared model cache.

The Python driver and decoded-program runner are shared with `../openpilot`.
They select vendor RKNPU or mainline Linux 6.18 Rocket from the device's sysfs
driver. `--driver rknpu` or `--driver rocket` requires that driver, and
`--device PATH` selects a specific NPU node. The same prepared cache is used.

On a mainline 6.18 board with Rocket enabled:

```sh
python3 -S gpt2/generate.py --driver rocket --prompt 'Hello' --tokens 5 --json
# Optional explicit device selection:
python3 -S gpt2/generate.py --driver rocket --device /dev/accel/accel0 --prompt 'Hello' --tokens 5
```

Rocket BO ownership, sequential task scheduling and PC trailer adaptation are
described in `../openpilot/README.md`. All transformer arithmetic stays on the
NPU. The results in `validation.json` were measured on vendor kernel 6.1.99,
RKNPU 0.9.8, core 0. The Rocket ABI and all four base-kernel schedules have
software regression coverage, but numerical verification on mainline hardware
is pending; see `../openpilot/MAINLINE.md`.

After preparing both model caches, `python3 -S openpilot/verify_models.py
--driver rocket --report /tmp/models-rocket.json` checks the three recorded
GPT-2 prompts and all three openpilot fixtures on the selected NPU driver.

## Preparation

These commands work from a fresh checkout. The cache is
`~/.cache/rk3588-models/gpt2` (about 930 MB, including the original 548 MB
checkpoint). Downloaded weights and expanded model buffers stay outside the
repository; readable decoded kernel templates ship in `templates/`.

```sh
python3 -S gpt2/setup.py
python3 -S gpt2/prepare.py
```

`setup.py` fetches the pinned `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and verifies file sizes and hashes.
`prepare.py` uses only the standard library to build all four base kernels from
the shipped templates and checkpoint, then packs the other eleven layers' own
weights into the verified layer-0 geometry. This includes the vocabulary
projection's 50,257 output rows and its compact final tile. Sparse constants
preserve the compiler's normalization/GELU tables. Every reconstructed base
buffer must match its verified SHA256. It also verifies complete layer-0
tensors before copying that layout and checks existing cache files on subsequent
runs. Weights are not shared across transformer layers.

To use another cache, pass `--cache DIRECTORY/checkpoint` to `setup.py` and
`--cache DIRECTORY` to both `prepare.py` and `generate.py`. Preparation needs
neither a pre-existing board cache nor an RKNN SDK. Initial setup needs internet
access; preparation and inference then run offline.

The four templates are `embedding`, `layer-00-qkv`,
`layer-00-attention-ffn`, and `head`. They retain the decoded registers from the
verified reference compilation. The original compiler, capture observer and
comparison fixtures are in `~/pilot/evaluation`:

- `compile_gpt2.py` exports the exact GPT-2 equations and uses RKNN Toolkit 2.3.2.
- `model_reference.py` captures named FP16 native inputs and reference outputs.
- `capture_memory.c` records complete allocations and decoded command streams.
- `openpilot/prepare_capture.py` converts those captures into relocatable data.
- `gpt2_baseline.py` runs the original Hugging Face FP32 model for comparison.

These tools are unnecessary for the shipped preparation path. For replacing a
kernel's geometry, compile its part externally, capture its real NPU
execution, and export the capture with `openpilot/prepare_capture.py`. Inference
never imports the compiler, RKNN, NumPy, PyTorch, or Transformers.

CNA matrices use output-channel-16/input-channel-32 coefficient tiles. Scale
vectors are FP16; bias vectors in these kernels are FP32. `prepare.py` contains
the verified transformer offsets and layouts; template tensor records specify
the base-kernel offsets. Runtime addresses are allocated anew and
relocated from named register fields. All kernel commands are reconstructed in
Python rather than submitted as a captured command blob.

## Validation and limits

The tokenizer matched Hugging Face on 300 reproducible mixed Unicode, control,
whitespace and punctuation samples. ASCII information separators are handled
separately from Unicode whitespace, matching GPT-2's regex behavior.

See `validation.json` for full measured results. Five continuation tokens
matched Hugging Face exactly for each of three prompts: `Hello`,
`The quick brown fox`, and `The capital of France is`. These ran in one
persistent decoder with resets between prompts; the last repeated an earlier
run to check cache reuse.

Component outputs match RKNN exactly. Python-packed layer-1 QKV matched an
independently compiled RKNN layer. Layer-1 attention/feed-forward packing also
matched an independent compiler when both kernels used the same 128-token
geometry. That check is separate from the shipped 1024-token decoder.

FP16 logits differ from the FP32 reference; matching greedy tokens on these
prompts does not establish identical probabilities or agreement on every
prompt. An exploratory comparison between 128- and 1024-token compiled kernels
failed exact output equality, so different compiled context sizes are not
interchangeable. Long-context accuracy and performance have not been measured.
