# GPT-2 124M on the RK3588 NPU

The requested inference runtime uses Python's standard library only. Tensor
arithmetic executes on the NPU; a CPU model is used only as a verification
reference in `/home/orangepi/pilot/evaluation`. Model weights live in an external
cache. The official checkpoint revision and file hashes are pinned in `setup.py`.

- [x] Fetch and verify the official checkpoint.
- [x] Implement checkpoint loading and tokenizer without package dependencies; tokenizer parity verified.
- [x] Prepare decoded NPU kernels and packed coefficients.
- [x] Implement transformer inference and KV caching.
- [x] Run complete text generation and compare logits/tokens with the reference.
- [x] Document verified commands and limitations.

## Observed outcome

The 1024-context full model runs with `python3 -S`. Independent component checks
match RKNN. Full greedy continuations match Hugging Face for Hello, the quick
brown fox, and the capital-of-France prompt (five tokens each). A persistent
decoder with resets repeated the capital prompt successfully. The standard
library weight packer was independently checked against compiler-produced QKV
and matching-size attention/feed-forward kernels. Re-running preparation
verified existing payloads.

FP16 logits differ from FP32, and a comparison across differently sized compiled
contexts failed exact equality. Long-context accuracy/performance remains
unmeasured. See `validation.json` and `README.md` for the measured scope.

The final tokenizer stress check found and fixed the Python/Unicode-regex
whitespace difference for ASCII information separators. All 300 reproducible
mixed Unicode/control/whitespace samples now match Hugging Face.

## Commit review fixes

All four base kernels now have readable decoded preparation templates in the
repository. `prepare.py` builds embedding addition, layer-0 QKV, layer-0
attention/FFN, and the vocabulary projection from the pinned checkpoint, then
packs the remaining eleven layers. Normalization/GELU constants are explicit
data. All four reconstructed base coefficient buffers match the verified
captures exactly, including the head's compact final output tile.

A new cache was built from actual public downloads without the original board
cache or SDK. An isolated source copy ran with `python3 -S -B`, empty
`PYTHONPATH`, and empty `LD_PRELOAD`. Five generated tokens matched the original
reference for Hello, the quick brown fox, and the capital-of-France prompt.
Repeat preparation verified all 26 kernels. Existing board caches also remain
compatible. Evidence is recorded in `validation.json` and
`~/pilot/evaluation/review-fixes/verification.json`.
