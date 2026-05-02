# conv.py — Remaining Problems

## Audit: Unreadable Or Special-Case Blocks

- **Weight layout dispatch (`_KH_MAJOR_SHAPES`)**: still a shape table keyed by `(out_c, in_c, kh, kw)` with group filters. This is the least principled part of the file because it encodes observed packing layouts rather than a decoded hardware field. Fixed by documenting it as a contained observed-layout table and keeping all dispatch in `_is_kh_major()`.
- **Small-channel pixel DMA selection (`in_channels in (1, 3, 4)`)**: RK3588/NVDLA image DMA accepts specific pixel component counts, while `ic=2` must stay off that path. Fixed by moving the tuple into named helpers/constants so the special case is readable and shared.
- **NHWC input packing decision**: formerly mixed shape facts, depthwise exclusion, and C2/channel ratio inline. Fixed by making depthwise exclusion explicit and using a named pack-C2 helper.
- **`ic=1` spatial convolution input pack C2**: `ic=1` spatial kernels use `c2=2` to trigger NHWC packing. This remains a hardware-format rule, but it is now isolated in `_input_pack_c2()`.
- **Output width stride exceptions**: generic output stride is derived from output atoms, but the `(in_c=3,out_c=6)` spatial cases still need observed strides. Fixed by routing through `_output_width_stride()` so remaining exceptions are localized.
- **Pixel-mode DMA line/surface stride branch**: feature mode and pixel mode use different line/surface stride formulas. Fixed by adding `_dma_strides()` instead of leaving inline branch math in `compute_conv2d_params()`.
- **CBUF entry and bank formulas**: correct but dense arithmetic. Fixed by moving into `_feature_grains()`, `_cbuf_entries()`, and `_data_bank()`.
- **Grouped convolution weight expansion**: grouped weights are expanded into dense channel slots before packing. Fixed by moving the inline transformation into `_expand_grouped_weights_fp16()`.
- **Depthwise packed weight expansion**: depthwise `CONV_MODE=3` fetches an aligned kernel footprint even for one-channel-per-group weights. Existing helper is retained with clearer naming around grouped expansion.
- **1x1 channel slicing for `in_channels >= 5`**: non-aligned pixel DMA only handles up to four channels, so larger pointwise convolutions are sliced into four-channel submits. Fixed by replacing literal `4` uses with `MAX_PIXEL_DMA_CHANNELS`.
- **Depthwise slicing above eight channels**: direct depthwise lanes are submitted in eight-channel chunks. Fixed by replacing literal `8` uses with `DEPTHWISE_SLICE_CHANNELS`.
- **Spatial convolution decomposition**: non-1x1 kernels are decomposed into exact-order 1x1 NPU submits to avoid direct spatial partial-output behavior. This remains a major workaround and is documented in the function docstring.
- **Single-input pointwise padding to three channels**: `ic=1` pointwise submit pads to three channels so it can use a supported pixel DMA format. Fixed by replacing literal `3` with `POINTWISE_PIXEL_CHANNELS`.
- **Raw register values in `build_conv2d_regs()`**: many register bitfields are still emitted as raw constants. This remains readable only with `experimental/rockchip.py` open; not fixed in this pass because replacing all bitfields risks changing working register programming.
- **Script entrypoint hard-codes one smoke shape**: useful for quick manual testing but not representative. Left as-is because `test/test_conv.py --submit` is the real coverage.

## Known Limitations

- **Cross-process isolation (P7)**: one bad NPU submission can corrupt state for later processes, even after `reset_npu()`. Recovery still requires reloading the kernel module.
- **Direct spatial convolution**: raw non-1x1 programming can still produce partial or shape-dependent output, so the tested path decomposes spatial kernels into 1x1 NPU submits.
