# Teflon Delegate Model Testing

## Prerequisites

mesa source code is at /home/orangepi/rk3588/ref/mesa

```bash
# from ~/mesa
source .venv/bin/activate

# install test image (if missing)
wget -O grace_hopper.bmp \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

# tflite delegate path
DELEGATE=build/src/gallium/targets/teflon/libteflon.so
```

## Models Available

| Model | Script | Teflon | Notes |
|-------|--------|--------|-------|
| mobilenetv1 | `classification.py` | yes | 224x224 uint8, 1001 classes |
| mobilenetv2 | `classification.py` | yes | 224x224 uint8, 1001 classes |
| inception | `classification.py` | yes | 224x224 uint8, 1001 classes |
| ssdmobilenetv2 | `detection.py` | yes | 300x300 uint8, COCO 80 classes |
| mobiledet | `detection.py` | yes | 320x320 uint8, COCO 80 classes |
| efficientdet | `detection.py` | no | 320x320 uint8, COCO — driver assertion bug |
| movenetlightning | `posenet.py` | no | 192x192 uint8, 17 keypoints — driver bug |
| movenetthunder | `posenet.py` | no | 256x256 uint8, 17 keypoints — driver bug |
| yolox | `yolox.py` | no | 416x416 int8, COCO — driver bug + needs letterbox |
| micronetlarge | — | — | 32x32 int8 grayscale, 8 outputs — audio model, not image |

## Test Scripts

All scripts accept `-e <delegate>` to enable the Teflon delegate (omit for CPU-only).

### Classification

Run from `~/mesa`, matching the command in `README.md`.

```bash
# mobilenetv1
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/classification.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/mobilenetv1/mobilenet_v1_1_224_quant.tflite \
  -l src/gallium/frontends/teflon/tests/labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE

# mobilenetv2
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/classification.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/mobilenetv2/mobilenet_v2_tflite_1_0_224_quantized_v1.tflite \
  -l src/gallium/frontends/teflon/tests/labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE

# inception
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/classification.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/inception/inception_v1_224_quant.tflite \
  -l src/gallium/frontends/teflon/tests/labels_mobilenet_quant_v1_224.txt \
  -e $DELEGATE
```

### Detection (SSD models with post-processing ops)

Uses `detection.py` (COCO labels built-in). Supports `--output` for annotated image and `--score_threshold` (default 0.5).

```bash
# ssdmobilenetv2 (works with teflon) — 300x300
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/detection.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/ssdmobilenetv2/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
  -e $DELEGATE \
  --output /tmp/ssd_out.bmp

# mobiledet (works with teflon) — 320x320
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 src/gallium/frontends/teflon/tests/detection.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/mobiledet/ssdlite_mobiledet_coco_qat_postprocess.tflite \
  -e $DELEGATE \
  --output /tmp/mobiledet_out.bmp

# efficientdet (CPU only — driver assertion bug)
python3.10 src/gallium/frontends/teflon/tests/detection.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/efficientdet/efficientdet_tflite_lite0_int8_v1.tflite \
  --output /tmp/efficientdet_out.bmp
```

### Pose Estimation

Uses `posenet.py` (17 keypoints, built-in skeleton edges). Supports `--output` and `--score_threshold`.

```bash
# movenetlightning (CPU only — driver assertion bug) — 192x192
python3.10 posenet.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/movenetlightning/movenet_single_pose_lightning_ptq.tflite \
  --output /tmp/movenet_lightning_out.bmp

# movenetthunder (CPU only — driver assertion bug) — 256x256
python3.10 posenet.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/movenetthunder/movenet_single_pose_thunder_ptq.tflite \
  --output /tmp/movenet_thunder_out.bmp
```

### YOLOX

Uses `yolox.py` (COCO labels, built-in NMS). Input is normalized to [0,1] then quantized per model parameters.

```bash
# CPU only — 416x416 int8
python3.10 yolox.py \
  -i ~/tensorflow/assets/grace_hopper.bmp \
  -m ../targets/teflon/tests/models/yolox/yolox_nano_full_integer_quant.tflite \
  --score_threshold 0.3 \
  --output /tmp/yolox_out.bmp
```

## Performance Results

| Model | Teflon | CPU-only | Quality |
|-------|--------|----------|---------|
| mobilenetv1 | ~11ms | — | military uniform 0.867 |
| mobilenetv2 | ~15ms | — | military uniform |
| inception | ~34ms | — | military uniform |
| ssdmobilenetv2 | ~27ms | — | 20 detections (low scores) |
| mobiledet | ~38ms | — | person 0.715, snowboard 0.676 |
| efficientdet | assertion | ~135ms | person 0.965 |
| movenetlightning | assertion | ~42ms | face/shoulders detected |
| movenetthunder | assertion | — | — |
| yolox | assertion | ~175ms | 0 detections @ 0.3 threshold |

## Extracted Mesa/Teflon Conv Shapes

Source: real `TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs ROCKET_DEBUG=dbg_msgs` compile logs from `~/mesa` for the example commands in this file.  `mobilenetv1` was rerun end-to-end from `~/mesa` with `./grace_hopper.bmp` and produced the expected `military uniform` result in about 11 ms; the other rows come from the same Teflon compile log format.

Shape notation follows Teflon tensor order: activations are `N x H x W x C`; weights are `OC x KH x KW x IC` for normal conv and `1 x KH x KW x C` for depthwise conv.  `conv.py exact` means the op can be represented as the current `experimental/mainline6_18/conv.py` stride-1 valid-convolution API without padding or stride.  Rows marked `no` are Mesa/TFLite quantized padded/strided/layout cases, not exact `conv.py` shapes.

| Model | Conv ops | CONV | DWCONV | Exact `conv.py` | Mesa-only shape semantics |
|-------|---------:|-----:|-------:|----------------:|--------------------------:|
| `mobilenetv1` | 28 | 15 | 13 | 14 | 14 |
| `mobilenetv2` | 53 | 36 | 17 | 35 | 18 |
| `inception` | 58 | 58 | 0 | 38 | 20 |
| `ssdmobilenetv2` | 72 | 55 | 17 | 51 | 21 |
| `mobiledet` | 93 | 66 | 27 | 54 | 39 |

### mobilenetv1

| Graph | Op | Type | Input NHWC | Weight OHWI | Output NHWC | Groups | `conv.py` exact |
|------:|---:|------|------------|-------------|-------------|-------:|-----------------|
| 0 | 0 | CONV | `1x224x224x3` | `32x3x3x3` | `1x112x112x32` | 1 | no: padding/stride or layout |
| 0 | 1 | DWCONV | `1x112x112x32` | `1x3x3x32` | `1x112x112x32` | 32 | no: padding/stride or layout |
| 0 | 2 | CONV | `1x112x112x32` | `64x1x1x32` | `1x112x112x64` | 1 | yes |
| 0 | 3 | DWCONV | `1x112x112x64` | `1x3x3x64` | `1x56x56x64` | 64 | no: padding/stride or layout |
| 0 | 4 | CONV | `1x56x56x64` | `128x1x1x64` | `1x56x56x128` | 1 | yes |
| 0 | 5 | DWCONV | `1x56x56x128` | `1x3x3x128` | `1x56x56x128` | 128 | no: padding/stride or layout |
| 0 | 6 | CONV | `1x56x56x128` | `128x1x1x128` | `1x56x56x128` | 1 | yes |
| 0 | 7 | DWCONV | `1x56x56x128` | `1x3x3x128` | `1x28x28x128` | 128 | no: padding/stride or layout |
| 0 | 8 | CONV | `1x28x28x128` | `256x1x1x128` | `1x28x28x256` | 1 | yes |
| 0 | 9 | DWCONV | `1x28x28x256` | `1x3x3x256` | `1x28x28x256` | 256 | no: padding/stride or layout |
| 0 | 10 | CONV | `1x28x28x256` | `256x1x1x256` | `1x28x28x256` | 1 | yes |
| 0 | 11 | DWCONV | `1x28x28x256` | `1x3x3x256` | `1x14x14x256` | 256 | no: padding/stride or layout |
| 0 | 12 | CONV | `1x14x14x256` | `512x1x1x256` | `1x14x14x512` | 1 | yes |
| 0 | 13 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x14x14x512` | 512 | no: padding/stride or layout |
| 0 | 14 | CONV | `1x14x14x512` | `512x1x1x512` | `1x14x14x512` | 1 | yes |
| 0 | 15 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x14x14x512` | 512 | no: padding/stride or layout |
| 0 | 16 | CONV | `1x14x14x512` | `512x1x1x512` | `1x14x14x512` | 1 | yes |
| 0 | 17 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x14x14x512` | 512 | no: padding/stride or layout |
| 0 | 18 | CONV | `1x14x14x512` | `512x1x1x512` | `1x14x14x512` | 1 | yes |
| 0 | 19 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x14x14x512` | 512 | no: padding/stride or layout |
| 0 | 20 | CONV | `1x14x14x512` | `512x1x1x512` | `1x14x14x512` | 1 | yes |
| 0 | 21 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x14x14x512` | 512 | no: padding/stride or layout |
| 0 | 22 | CONV | `1x14x14x512` | `512x1x1x512` | `1x14x14x512` | 1 | yes |
| 0 | 23 | DWCONV | `1x14x14x512` | `1x3x3x512` | `1x7x7x512` | 512 | no: padding/stride or layout |
| 0 | 24 | CONV | `1x7x7x512` | `1024x1x1x512` | `1x7x7x1024` | 1 | yes |
| 0 | 25 | DWCONV | `1x7x7x1024` | `1x3x3x1024` | `1x7x7x1024` | 1024 | no: padding/stride or layout |
| 0 | 26 | CONV | `1x7x7x1024` | `1024x1x1x1024` | `1x7x7x1024` | 1 | yes |
| 1 | 0 | CONV | `1x1x1x1024` | `1001x1x1x1024` | `1x1x1x1001` | 1 | yes |

### mobilenetv2

| Graph | Op | Type | Input NHWC | Weight OHWI | Output NHWC | Groups | `conv.py` exact |
|------:|---:|------|------------|-------------|-------------|-------:|-----------------|
| 0 | 0 | CONV | `1x224x224x3` | `32x3x3x3` | `1x112x112x32` | 1 | no: padding/stride or layout |
| 0 | 1 | DWCONV | `1x112x112x32` | `1x3x3x32` | `1x112x112x32` | 32 | no: padding/stride or layout |
| 0 | 2 | CONV | `1x112x112x32` | `16x1x1x32` | `1x112x112x16` | 1 | yes |
| 0 | 3 | CONV | `1x112x112x16` | `96x1x1x16` | `1x112x112x96` | 1 | yes |
| 0 | 4 | DWCONV | `1x112x112x96` | `1x3x3x96` | `1x56x56x96` | 96 | no: padding/stride or layout |
| 0 | 5 | CONV | `1x56x56x96` | `24x1x1x96` | `1x56x56x24` | 1 | yes |
| 0 | 6 | CONV | `1x56x56x24` | `144x1x1x24` | `1x56x56x144` | 1 | yes |
| 0 | 7 | DWCONV | `1x56x56x144` | `1x3x3x144` | `1x56x56x144` | 144 | no: padding/stride or layout |
| 0 | 8 | CONV | `1x56x56x144` | `24x1x1x144` | `1x56x56x24` | 1 | yes |
| 0 | 10 | CONV | `1x56x56x24` | `144x1x1x24` | `1x56x56x144` | 1 | yes |
| 0 | 11 | DWCONV | `1x56x56x144` | `1x3x3x144` | `1x28x28x144` | 144 | no: padding/stride or layout |
| 0 | 12 | CONV | `1x28x28x144` | `32x1x1x144` | `1x28x28x32` | 1 | yes |
| 0 | 13 | CONV | `1x28x28x32` | `192x1x1x32` | `1x28x28x192` | 1 | yes |
| 0 | 14 | DWCONV | `1x28x28x192` | `1x3x3x192` | `1x28x28x192` | 192 | no: padding/stride or layout |
| 0 | 15 | CONV | `1x28x28x192` | `32x1x1x192` | `1x28x28x32` | 1 | yes |
| 0 | 17 | CONV | `1x28x28x32` | `192x1x1x32` | `1x28x28x192` | 1 | yes |
| 0 | 18 | DWCONV | `1x28x28x192` | `1x3x3x192` | `1x28x28x192` | 192 | no: padding/stride or layout |
| 0 | 19 | CONV | `1x28x28x192` | `32x1x1x192` | `1x28x28x32` | 1 | yes |
| 0 | 21 | CONV | `1x28x28x32` | `192x1x1x32` | `1x28x28x192` | 1 | yes |
| 0 | 22 | DWCONV | `1x28x28x192` | `1x3x3x192` | `1x14x14x192` | 192 | no: padding/stride or layout |
| 0 | 23 | CONV | `1x14x14x192` | `64x1x1x192` | `1x14x14x64` | 1 | yes |
| 0 | 24 | CONV | `1x14x14x64` | `384x1x1x64` | `1x14x14x384` | 1 | yes |
| 0 | 25 | DWCONV | `1x14x14x384` | `1x3x3x384` | `1x14x14x384` | 384 | no: padding/stride or layout |
| 0 | 26 | CONV | `1x14x14x384` | `64x1x1x384` | `1x14x14x64` | 1 | yes |
| 0 | 28 | CONV | `1x14x14x64` | `384x1x1x64` | `1x14x14x384` | 1 | yes |
| 0 | 29 | DWCONV | `1x14x14x384` | `1x3x3x384` | `1x14x14x384` | 384 | no: padding/stride or layout |
| 0 | 30 | CONV | `1x14x14x384` | `64x1x1x384` | `1x14x14x64` | 1 | yes |
| 0 | 32 | CONV | `1x14x14x64` | `384x1x1x64` | `1x14x14x384` | 1 | yes |
| 0 | 33 | DWCONV | `1x14x14x384` | `1x3x3x384` | `1x14x14x384` | 384 | no: padding/stride or layout |
| 0 | 34 | CONV | `1x14x14x384` | `64x1x1x384` | `1x14x14x64` | 1 | yes |
| 0 | 36 | CONV | `1x14x14x64` | `384x1x1x64` | `1x14x14x384` | 1 | yes |
| 0 | 37 | DWCONV | `1x14x14x384` | `1x3x3x384` | `1x14x14x384` | 384 | no: padding/stride or layout |
| 0 | 38 | CONV | `1x14x14x384` | `96x1x1x384` | `1x14x14x96` | 1 | yes |
| 0 | 39 | CONV | `1x14x14x96` | `576x1x1x96` | `1x14x14x576` | 1 | yes |
| 0 | 40 | DWCONV | `1x14x14x576` | `1x3x3x576` | `1x14x14x576` | 576 | no: padding/stride or layout |
| 0 | 41 | CONV | `1x14x14x576` | `96x1x1x576` | `1x14x14x96` | 1 | yes |
| 0 | 43 | CONV | `1x14x14x96` | `576x1x1x96` | `1x14x14x576` | 1 | yes |
| 0 | 44 | DWCONV | `1x14x14x576` | `1x3x3x576` | `1x14x14x576` | 576 | no: padding/stride or layout |
| 0 | 45 | CONV | `1x14x14x576` | `96x1x1x576` | `1x14x14x96` | 1 | yes |
| 0 | 47 | CONV | `1x14x14x96` | `576x1x1x96` | `1x14x14x576` | 1 | yes |
| 0 | 48 | DWCONV | `1x14x14x576` | `1x3x3x576` | `1x7x7x576` | 576 | no: padding/stride or layout |
| 0 | 49 | CONV | `1x7x7x576` | `160x1x1x576` | `1x7x7x160` | 1 | yes |
| 0 | 50 | CONV | `1x7x7x160` | `960x1x1x160` | `1x7x7x960` | 1 | yes |
| 0 | 51 | DWCONV | `1x7x7x960` | `1x3x3x960` | `1x7x7x960` | 960 | no: padding/stride or layout |
| 0 | 52 | CONV | `1x7x7x960` | `160x1x1x960` | `1x7x7x160` | 1 | yes |
| 0 | 54 | CONV | `1x7x7x160` | `960x1x1x160` | `1x7x7x960` | 1 | yes |
| 0 | 55 | DWCONV | `1x7x7x960` | `1x3x3x960` | `1x7x7x960` | 960 | no: padding/stride or layout |
| 0 | 56 | CONV | `1x7x7x960` | `160x1x1x960` | `1x7x7x160` | 1 | yes |
| 0 | 58 | CONV | `1x7x7x160` | `960x1x1x160` | `1x7x7x960` | 1 | yes |
| 0 | 59 | DWCONV | `1x7x7x960` | `1x3x3x960` | `1x7x7x960` | 960 | no: padding/stride or layout |
| 0 | 60 | CONV | `1x7x7x960` | `320x1x1x960` | `1x7x7x320` | 1 | yes |
| 0 | 61 | CONV | `1x7x7x320` | `1280x1x1x320` | `1x7x7x1280` | 1 | yes |
| 1 | 0 | CONV | `1x1x1x1280` | `1001x1x1x1280` | `1x1x1x1001` | 1 | yes |

### inception

| Graph | Op | Type | Input NHWC | Weight OHWI | Output NHWC | Groups | `conv.py` exact |
|------:|---:|------|------------|-------------|-------------|-------:|-----------------|
| 0 | 0 | CONV | `1x224x224x3` | `64x7x7x3` | `1x112x112x64` | 1 | no: padding/stride or layout |
| 1 | 0 | CONV | `1x56x56x64` | `64x1x1x64` | `1x56x56x64` | 1 | yes |
| 1 | 1 | CONV | `1x56x56x64` | `192x3x3x64` | `1x56x56x192` | 1 | no: padding/stride or layout |
| 2 | 0 | CONV | `1x28x28x192` | `64x1x1x192` | `1x28x28x64` | 1 | yes |
| 2 | 1 | CONV | `1x28x28x192` | `96x1x1x192` | `1x28x28x96` | 1 | yes |
| 2 | 2 | CONV | `1x28x28x96` | `128x3x3x96` | `1x28x28x128` | 1 | no: padding/stride or layout |
| 2 | 3 | CONV | `1x28x28x192` | `16x1x1x192` | `1x28x28x16` | 1 | yes |
| 2 | 4 | CONV | `1x28x28x16` | `32x3x3x16` | `1x28x28x32` | 1 | no: padding/stride or layout |
| 2 | 5 | CONV | `1x28x28x192` | `32x1x1x192` | `1x28x28x32` | 1 | yes |
| 3 | 0 | CONV | `1x28x28x256` | `128x1x1x256` | `1x28x28x128` | 1 | yes |
| 3 | 1 | CONV | `1x28x28x256` | `128x1x1x256` | `1x28x28x128` | 1 | yes |
| 3 | 2 | CONV | `1x28x28x128` | `192x3x3x128` | `1x28x28x192` | 1 | no: padding/stride or layout |
| 3 | 3 | CONV | `1x28x28x256` | `32x1x1x256` | `1x28x28x32` | 1 | yes |
| 3 | 4 | CONV | `1x28x28x32` | `96x3x3x32` | `1x28x28x96` | 1 | no: padding/stride or layout |
| 3 | 5 | CONV | `1x28x28x256` | `64x1x1x256` | `1x28x28x64` | 1 | yes |
| 4 | 0 | CONV | `1x14x14x480` | `192x1x1x480` | `1x14x14x192` | 1 | yes |
| 4 | 1 | CONV | `1x14x14x480` | `96x1x1x480` | `1x14x14x96` | 1 | yes |
| 4 | 2 | CONV | `1x14x14x96` | `208x3x3x96` | `1x14x14x208` | 1 | no: padding/stride or layout |
| 4 | 3 | CONV | `1x14x14x480` | `16x1x1x480` | `1x14x14x16` | 1 | yes |
| 4 | 4 | CONV | `1x14x14x16` | `48x3x3x16` | `1x14x14x48` | 1 | no: padding/stride or layout |
| 4 | 5 | CONV | `1x14x14x480` | `64x1x1x480` | `1x14x14x64` | 1 | yes |
| 5 | 0 | CONV | `1x14x14x512` | `160x1x1x512` | `1x14x14x160` | 1 | yes |
| 5 | 1 | CONV | `1x14x14x512` | `112x1x1x512` | `1x14x14x112` | 1 | yes |
| 5 | 2 | CONV | `1x14x14x112` | `224x3x3x112` | `1x14x14x224` | 1 | no: padding/stride or layout |
| 5 | 3 | CONV | `1x14x14x512` | `24x1x1x512` | `1x14x14x24` | 1 | yes |
| 5 | 4 | CONV | `1x14x14x24` | `64x3x3x24` | `1x14x14x64` | 1 | no: padding/stride or layout |
| 5 | 5 | CONV | `1x14x14x512` | `64x1x1x512` | `1x14x14x64` | 1 | yes |
| 6 | 0 | CONV | `1x14x14x512` | `128x1x1x512` | `1x14x14x128` | 1 | yes |
| 6 | 1 | CONV | `1x14x14x512` | `128x1x1x512` | `1x14x14x128` | 1 | yes |
| 6 | 2 | CONV | `1x14x14x128` | `256x3x3x128` | `1x14x14x256` | 1 | no: padding/stride or layout |
| 6 | 3 | CONV | `1x14x14x512` | `24x1x1x512` | `1x14x14x24` | 1 | yes |
| 6 | 4 | CONV | `1x14x14x24` | `64x3x3x24` | `1x14x14x64` | 1 | no: padding/stride or layout |
| 6 | 5 | CONV | `1x14x14x512` | `64x1x1x512` | `1x14x14x64` | 1 | yes |
| 7 | 0 | CONV | `1x14x14x512` | `112x1x1x512` | `1x14x14x112` | 1 | yes |
| 7 | 1 | CONV | `1x14x14x512` | `144x1x1x512` | `1x14x14x144` | 1 | yes |
| 7 | 2 | CONV | `1x14x14x144` | `288x3x3x144` | `1x14x14x288` | 1 | no: padding/stride or layout |
| 7 | 3 | CONV | `1x14x14x512` | `32x1x1x512` | `1x14x14x32` | 1 | yes |
| 7 | 4 | CONV | `1x14x14x32` | `64x3x3x32` | `1x14x14x64` | 1 | no: padding/stride or layout |
| 7 | 5 | CONV | `1x14x14x512` | `64x1x1x512` | `1x14x14x64` | 1 | yes |
| 8 | 0 | CONV | `1x14x14x528` | `256x1x1x528` | `1x14x14x256` | 1 | yes |
| 8 | 1 | CONV | `1x14x14x528` | `160x1x1x528` | `1x14x14x160` | 1 | yes |
| 8 | 2 | CONV | `1x14x14x160` | `320x3x3x160` | `1x14x14x320` | 1 | no: padding/stride or layout |
| 8 | 3 | CONV | `1x14x14x528` | `32x1x1x528` | `1x14x14x32` | 1 | yes |
| 8 | 4 | CONV | `1x14x14x32` | `128x3x3x32` | `1x14x14x128` | 1 | no: padding/stride or layout |
| 8 | 5 | CONV | `1x14x14x528` | `128x1x1x528` | `1x14x14x128` | 1 | yes |
| 9 | 0 | CONV | `1x7x7x832` | `256x1x1x832` | `1x7x7x256` | 1 | yes |
| 9 | 1 | CONV | `1x7x7x832` | `160x1x1x832` | `1x7x7x160` | 1 | yes |
| 9 | 2 | CONV | `1x7x7x160` | `320x3x3x160` | `1x7x7x320` | 1 | no: padding/stride or layout |
| 9 | 3 | CONV | `1x7x7x832` | `32x1x1x832` | `1x7x7x32` | 1 | yes |
| 9 | 4 | CONV | `1x7x7x32` | `128x3x3x32` | `1x7x7x128` | 1 | no: padding/stride or layout |
| 9 | 5 | CONV | `1x7x7x832` | `128x1x1x832` | `1x7x7x128` | 1 | yes |
| 10 | 0 | CONV | `1x7x7x832` | `384x1x1x832` | `1x7x7x384` | 1 | yes |
| 10 | 1 | CONV | `1x7x7x832` | `192x1x1x832` | `1x7x7x192` | 1 | yes |
| 10 | 2 | CONV | `1x7x7x192` | `384x3x3x192` | `1x7x7x384` | 1 | no: padding/stride or layout |
| 10 | 3 | CONV | `1x7x7x832` | `48x1x1x832` | `1x7x7x48` | 1 | yes |
| 10 | 4 | CONV | `1x7x7x48` | `128x3x3x48` | `1x7x7x128` | 1 | no: padding/stride or layout |
| 10 | 5 | CONV | `1x7x7x832` | `128x1x1x832` | `1x7x7x128` | 1 | yes |
| 11 | 0 | CONV | `1x1x1x1024` | `1001x1x1x1024` | `1x1x1x1001` | 1 | yes |

### ssdmobilenetv2

| Graph | Op | Type | Input NHWC | Weight OHWI | Output NHWC | Groups | `conv.py` exact |
|------:|---:|------|------------|-------------|-------------|-------:|-----------------|
| 0 | 0 | CONV | `0x0x0x0` | `32x3x3x3` | `1x150x150x32` | 1 | no: dynamic/zero dim |
| 0 | 1 | DWCONV | `1x150x150x32` | `1x3x3x32` | `1x150x150x32` | 32 | no: padding/stride or layout |
| 0 | 2 | CONV | `1x150x150x32` | `16x1x1x32` | `1x150x150x16` | 1 | yes |
| 0 | 3 | CONV | `1x150x150x16` | `96x1x1x16` | `1x150x150x96` | 1 | yes |
| 0 | 4 | DWCONV | `1x150x150x96` | `1x3x3x96` | `1x75x75x96` | 96 | no: padding/stride or layout |
| 0 | 5 | CONV | `1x75x75x96` | `24x1x1x96` | `1x75x75x24` | 1 | yes |
| 0 | 6 | CONV | `1x75x75x24` | `144x1x1x24` | `1x75x75x144` | 1 | yes |
| 0 | 7 | DWCONV | `1x75x75x144` | `1x3x3x144` | `1x75x75x144` | 144 | no: padding/stride or layout |
| 0 | 8 | CONV | `1x75x75x144` | `24x1x1x144` | `1x75x75x24` | 1 | yes |
| 0 | 10 | CONV | `1x75x75x24` | `144x1x1x24` | `1x75x75x144` | 1 | yes |
| 0 | 11 | DWCONV | `1x75x75x144` | `1x3x3x144` | `1x38x38x144` | 144 | no: padding/stride or layout |
| 0 | 12 | CONV | `1x38x38x144` | `32x1x1x144` | `1x38x38x32` | 1 | yes |
| 0 | 13 | CONV | `1x38x38x32` | `192x1x1x32` | `1x38x38x192` | 1 | yes |
| 0 | 14 | DWCONV | `1x38x38x192` | `1x3x3x192` | `1x38x38x192` | 192 | no: padding/stride or layout |
| 0 | 15 | CONV | `1x38x38x192` | `32x1x1x192` | `1x38x38x32` | 1 | yes |
| 0 | 17 | CONV | `1x38x38x32` | `192x1x1x32` | `1x38x38x192` | 1 | yes |
| 0 | 18 | DWCONV | `1x38x38x192` | `1x3x3x192` | `1x38x38x192` | 192 | no: padding/stride or layout |
| 0 | 19 | CONV | `1x38x38x192` | `32x1x1x192` | `1x38x38x32` | 1 | yes |
| 0 | 21 | CONV | `1x38x38x32` | `192x1x1x32` | `1x38x38x192` | 1 | yes |
| 0 | 22 | DWCONV | `1x38x38x192` | `1x3x3x192` | `1x19x19x192` | 192 | no: padding/stride or layout |
| 0 | 23 | CONV | `1x19x19x192` | `64x1x1x192` | `1x19x19x64` | 1 | yes |
| 0 | 24 | CONV | `1x19x19x64` | `384x1x1x64` | `1x19x19x384` | 1 | yes |
| 0 | 25 | DWCONV | `1x19x19x384` | `1x3x3x384` | `1x19x19x384` | 384 | no: padding/stride or layout |
| 0 | 26 | CONV | `1x19x19x384` | `64x1x1x384` | `1x19x19x64` | 1 | yes |
| 0 | 28 | CONV | `1x19x19x64` | `384x1x1x64` | `1x19x19x384` | 1 | yes |
| 0 | 29 | DWCONV | `1x19x19x384` | `1x3x3x384` | `1x19x19x384` | 384 | no: padding/stride or layout |
| 0 | 30 | CONV | `1x19x19x384` | `64x1x1x384` | `1x19x19x64` | 1 | yes |
| 0 | 32 | CONV | `1x19x19x64` | `384x1x1x64` | `1x19x19x384` | 1 | yes |
| 0 | 33 | DWCONV | `1x19x19x384` | `1x3x3x384` | `1x19x19x384` | 384 | no: padding/stride or layout |
| 0 | 34 | CONV | `1x19x19x384` | `64x1x1x384` | `1x19x19x64` | 1 | yes |
| 0 | 36 | CONV | `1x19x19x64` | `384x1x1x64` | `1x19x19x384` | 1 | yes |
| 0 | 37 | DWCONV | `1x19x19x384` | `1x3x3x384` | `1x19x19x384` | 384 | no: padding/stride or layout |
| 0 | 38 | CONV | `1x19x19x384` | `96x1x1x384` | `1x19x19x96` | 1 | yes |
| 0 | 39 | CONV | `1x19x19x96` | `576x1x1x96` | `1x19x19x576` | 1 | yes |
| 0 | 40 | DWCONV | `1x19x19x576` | `1x3x3x576` | `1x19x19x576` | 576 | no: padding/stride or layout |
| 0 | 41 | CONV | `1x19x19x576` | `96x1x1x576` | `1x19x19x96` | 1 | yes |
| 0 | 43 | CONV | `1x19x19x96` | `576x1x1x96` | `1x19x19x576` | 1 | yes |
| 0 | 44 | DWCONV | `1x19x19x576` | `1x3x3x576` | `1x19x19x576` | 576 | no: padding/stride or layout |
| 0 | 45 | CONV | `1x19x19x576` | `96x1x1x576` | `1x19x19x96` | 1 | yes |
| 0 | 47 | CONV | `1x19x19x96` | `576x1x1x96` | `1x19x19x576` | 1 | yes |
| 0 | 48 | CONV | `1x19x19x576` | `12x1x1x576` | `1x19x19x12` | 1 | yes |
| 0 | 49 | CONV | `1x19x19x576` | `273x1x1x576` | `1x19x19x273` | 1 | yes |
| 0 | 50 | DWCONV | `1x19x19x576` | `1x3x3x576` | `1x10x10x576` | 576 | no: padding/stride or layout |
| 0 | 51 | CONV | `1x10x10x576` | `160x1x1x576` | `1x10x10x160` | 1 | yes |
| 0 | 52 | CONV | `1x10x10x160` | `960x1x1x160` | `1x10x10x960` | 1 | yes |
| 0 | 53 | DWCONV | `1x10x10x960` | `1x3x3x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 54 | CONV | `1x10x10x960` | `160x1x1x960` | `1x10x10x160` | 1 | yes |
| 0 | 56 | CONV | `1x10x10x160` | `960x1x1x160` | `1x10x10x960` | 1 | yes |
| 0 | 57 | DWCONV | `1x10x10x960` | `1x3x3x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 58 | CONV | `1x10x10x960` | `160x1x1x960` | `1x10x10x160` | 1 | yes |
| 0 | 60 | CONV | `1x10x10x160` | `960x1x1x160` | `1x10x10x960` | 1 | yes |
| 0 | 61 | DWCONV | `1x10x10x960` | `1x3x3x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 62 | CONV | `1x10x10x960` | `320x1x1x960` | `1x10x10x320` | 1 | yes |
| 0 | 63 | CONV | `1x10x10x320` | `1280x1x1x320` | `1x10x10x1280` | 1 | yes |
| 0 | 64 | CONV | `1x10x10x1280` | `24x1x1x1280` | `1x10x10x24` | 1 | yes |
| 0 | 65 | CONV | `1x10x10x1280` | `546x1x1x1280` | `1x10x10x546` | 1 | yes |
| 0 | 66 | CONV | `1x10x10x1280` | `256x1x1x1280` | `1x10x10x256` | 1 | yes |
| 0 | 67 | CONV | `1x10x10x256` | `512x3x3x256` | `1x5x5x512` | 1 | no: padding/stride or layout |
| 0 | 68 | CONV | `1x5x5x512` | `24x1x1x512` | `1x5x5x24` | 1 | yes |
| 0 | 69 | CONV | `1x5x5x512` | `546x1x1x512` | `1x5x5x546` | 1 | yes |
| 0 | 70 | CONV | `1x5x5x512` | `128x1x1x512` | `1x5x5x128` | 1 | yes |
| 0 | 71 | CONV | `1x5x5x128` | `256x3x3x128` | `1x3x3x256` | 1 | yes |
| 0 | 72 | CONV | `1x3x3x256` | `24x1x1x256` | `1x3x3x24` | 1 | yes |
| 0 | 73 | CONV | `1x3x3x256` | `546x1x1x256` | `1x3x3x546` | 1 | yes |
| 0 | 74 | CONV | `1x3x3x256` | `128x1x1x256` | `1x3x3x128` | 1 | yes |
| 0 | 75 | CONV | `1x3x3x128` | `256x3x3x128` | `1x2x2x256` | 1 | no: padding/stride or layout |
| 0 | 76 | CONV | `1x2x2x256` | `24x1x1x256` | `1x2x2x24` | 1 | yes |
| 0 | 77 | CONV | `1x2x2x256` | `546x1x1x256` | `1x2x2x546` | 1 | yes |
| 0 | 78 | CONV | `1x2x2x256` | `64x1x1x256` | `1x2x2x64` | 1 | yes |
| 0 | 79 | CONV | `1x2x2x64` | `128x3x3x64` | `1x1x1x128` | 1 | no: padding/stride or layout |
| 0 | 80 | CONV | `1x1x1x128` | `24x1x1x128` | `1x1x1x24` | 1 | yes |
| 0 | 81 | CONV | `1x1x1x128` | `546x1x1x128` | `1x1x1x546` | 1 | yes |

### mobiledet

| Graph | Op | Type | Input NHWC | Weight OHWI | Output NHWC | Groups | `conv.py` exact |
|------:|---:|------|------------|-------------|-------------|-------:|-----------------|
| 0 | 0 | CONV | `1x320x320x3` | `32x3x3x3` | `1x160x160x32` | 1 | no: padding/stride or layout |
| 0 | 1 | CONV | `1x160x160x32` | `8x1x1x32` | `1x160x160x8` | 1 | yes |
| 0 | 2 | CONV | `1x160x160x8` | `16x3x3x8` | `1x160x160x16` | 1 | no: padding/stride or layout |
| 0 | 3 | CONV | `1x160x160x16` | `16x1x1x16` | `1x160x160x16` | 1 | yes |
| 0 | 4 | CONV | `1x160x160x16` | `128x3x3x16` | `1x80x80x128` | 1 | no: padding/stride or layout |
| 0 | 5 | CONV | `1x80x80x128` | `16x1x1x128` | `1x80x80x16` | 1 | yes |
| 0 | 6 | CONV | `1x80x80x16` | `64x3x3x16` | `1x80x80x64` | 1 | no: padding/stride or layout |
| 0 | 7 | CONV | `1x80x80x64` | `16x1x1x64` | `1x80x80x16` | 1 | yes |
| 0 | 9 | CONV | `1x80x80x16` | `128x3x3x16` | `1x80x80x128` | 1 | no: padding/stride or layout |
| 0 | 10 | CONV | `1x80x80x128` | `16x1x1x128` | `1x80x80x16` | 1 | yes |
| 0 | 12 | CONV | `1x80x80x16` | `64x3x3x16` | `1x80x80x64` | 1 | no: padding/stride or layout |
| 0 | 13 | CONV | `1x80x80x64` | `16x1x1x64` | `1x80x80x16` | 1 | yes |
| 0 | 15 | CONV | `1x80x80x16` | `128x5x5x16` | `1x40x40x128` | 1 | no: padding/stride or layout |
| 0 | 16 | CONV | `1x40x40x128` | `40x1x1x128` | `1x40x40x40` | 1 | yes |
| 0 | 17 | CONV | `1x40x40x40` | `160x3x3x40` | `1x40x40x160` | 1 | no: padding/stride or layout |
| 0 | 18 | CONV | `1x40x40x160` | `40x1x1x160` | `1x40x40x40` | 1 | yes |
| 0 | 20 | CONV | `1x40x40x40` | `160x3x3x40` | `1x40x40x160` | 1 | no: padding/stride or layout |
| 0 | 21 | CONV | `1x40x40x160` | `40x1x1x160` | `1x40x40x40` | 1 | yes |
| 0 | 23 | CONV | `1x40x40x40` | `160x3x3x40` | `1x40x40x160` | 1 | no: padding/stride or layout |
| 0 | 24 | CONV | `1x40x40x160` | `40x1x1x160` | `1x40x40x40` | 1 | yes |
| 0 | 26 | CONV | `1x40x40x40` | `320x1x1x40` | `1x40x40x320` | 1 | yes |
| 0 | 27 | DWCONV | `1x40x40x320` | `1x3x3x320` | `1x20x20x320` | 320 | no: padding/stride or layout |
| 0 | 28 | CONV | `1x20x20x320` | `72x1x1x320` | `1x20x20x72` | 1 | yes |
| 0 | 29 | CONV | `1x20x20x72` | `576x1x1x72` | `1x20x20x576` | 1 | yes |
| 0 | 30 | DWCONV | `1x20x20x576` | `1x3x3x576` | `1x20x20x576` | 576 | no: padding/stride or layout |
| 0 | 31 | CONV | `1x20x20x576` | `72x1x1x576` | `1x20x20x72` | 1 | yes |
| 0 | 33 | CONV | `1x20x20x72` | `288x3x3x72` | `1x20x20x288` | 1 | no: padding/stride or layout |
| 0 | 34 | CONV | `1x20x20x288` | `72x1x1x288` | `1x20x20x72` | 1 | yes |
| 0 | 36 | CONV | `1x20x20x72` | `288x3x3x72` | `1x20x20x288` | 1 | no: padding/stride or layout |
| 0 | 37 | CONV | `1x20x20x288` | `72x1x1x288` | `1x20x20x72` | 1 | yes |
| 0 | 39 | CONV | `1x20x20x72` | `576x1x1x72` | `1x20x20x576` | 1 | yes |
| 0 | 40 | DWCONV | `1x20x20x576` | `1x5x5x576` | `1x20x20x576` | 576 | no: padding/stride or layout |
| 0 | 41 | CONV | `1x20x20x576` | `96x1x1x576` | `1x20x20x96` | 1 | yes |
| 0 | 42 | CONV | `1x20x20x96` | `768x1x1x96` | `1x20x20x768` | 1 | yes |
| 0 | 43 | DWCONV | `1x20x20x768` | `1x5x5x768` | `1x20x20x768` | 768 | no: padding/stride or layout |
| 0 | 44 | CONV | `1x20x20x768` | `96x1x1x768` | `1x20x20x96` | 1 | yes |
| 0 | 46 | CONV | `1x20x20x96` | `768x1x1x96` | `1x20x20x768` | 1 | yes |
| 0 | 47 | DWCONV | `1x20x20x768` | `1x3x3x768` | `1x20x20x768` | 768 | no: padding/stride or layout |
| 0 | 48 | CONV | `1x20x20x768` | `96x1x1x768` | `1x20x20x96` | 1 | yes |
| 0 | 50 | CONV | `1x20x20x96` | `768x1x1x96` | `1x20x20x768` | 1 | yes |
| 0 | 51 | DWCONV | `1x20x20x768` | `1x3x3x768` | `1x20x20x768` | 768 | no: padding/stride or layout |
| 0 | 52 | CONV | `1x20x20x768` | `96x1x1x768` | `1x20x20x96` | 1 | yes |
| 0 | 54 | CONV | `1x20x20x96` | `768x1x1x96` | `1x20x20x768` | 1 | yes |
| 0 | 55 | DWCONV | `1x20x20x768` | `1x5x5x768` | `1x10x10x768` | 768 | no: padding/stride or layout |
| 0 | 56 | CONV | `1x10x10x768` | `120x1x1x768` | `1x10x10x120` | 1 | yes |
| 0 | 57 | CONV | `1x10x10x120` | `960x1x1x120` | `1x10x10x960` | 1 | yes |
| 0 | 58 | DWCONV | `1x10x10x960` | `1x3x3x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 59 | CONV | `1x10x10x960` | `120x1x1x960` | `1x10x10x120` | 1 | yes |
| 0 | 61 | CONV | `1x10x10x120` | `480x1x1x120` | `1x10x10x480` | 1 | yes |
| 0 | 62 | DWCONV | `1x10x10x480` | `1x5x5x480` | `1x10x10x480` | 480 | no: padding/stride or layout |
| 0 | 63 | CONV | `1x10x10x480` | `120x1x1x480` | `1x10x10x120` | 1 | yes |
| 0 | 65 | CONV | `1x10x10x120` | `960x1x1x120` | `1x10x10x960` | 1 | yes |
| 0 | 66 | DWCONV | `1x10x10x960` | `1x3x3x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 67 | CONV | `1x10x10x960` | `120x1x1x960` | `1x10x10x120` | 1 | yes |
| 0 | 69 | CONV | `1x10x10x120` | `960x1x1x120` | `1x10x10x960` | 1 | yes |
| 0 | 70 | DWCONV | `1x10x10x960` | `1x5x5x960` | `1x10x10x960` | 960 | no: padding/stride or layout |
| 0 | 71 | CONV | `1x10x10x960` | `384x1x1x960` | `1x10x10x384` | 1 | yes |
| 0 | 72 | CONV | `1x10x10x384` | `256x1x1x384` | `1x10x10x256` | 1 | yes |
| 0 | 73 | DWCONV | `1x10x10x256` | `1x3x3x256` | `1x5x5x256` | 256 | no: padding/stride or layout |
| 0 | 74 | CONV | `1x5x5x256` | `512x1x1x256` | `1x5x5x512` | 1 | yes |
| 0 | 75 | CONV | `1x5x5x512` | `128x1x1x512` | `1x5x5x128` | 1 | yes |
| 0 | 76 | DWCONV | `1x5x5x128` | `1x3x3x128` | `1x3x3x128` | 128 | yes |
| 0 | 77 | CONV | `1x3x3x128` | `256x1x1x128` | `1x3x3x256` | 1 | yes |
| 0 | 78 | CONV | `1x3x3x256` | `128x1x1x256` | `1x3x3x128` | 1 | yes |
| 0 | 79 | DWCONV | `1x3x3x128` | `1x3x3x128` | `1x2x2x128` | 128 | no: padding/stride or layout |
| 0 | 80 | CONV | `1x2x2x128` | `256x1x1x128` | `1x2x2x256` | 1 | yes |
| 0 | 81 | CONV | `1x2x2x256` | `64x1x1x256` | `1x2x2x64` | 1 | yes |
| 0 | 82 | DWCONV | `1x2x2x64` | `1x3x3x64` | `1x1x1x64` | 64 | no: padding/stride or layout |
| 0 | 83 | CONV | `1x1x1x64` | `128x1x1x64` | `1x1x1x128` | 1 | yes |
| 0 | 84 | DWCONV | `1x20x20x96` | `1x3x3x96` | `1x20x20x96` | 96 | no: padding/stride or layout |
| 0 | 85 | DWCONV | `1x20x20x96` | `1x3x3x96` | `1x20x20x96` | 96 | no: padding/stride or layout |
| 0 | 86 | DWCONV | `1x10x10x384` | `1x3x3x384` | `1x10x10x384` | 384 | no: padding/stride or layout |
| 0 | 87 | DWCONV | `1x10x10x384` | `1x3x3x384` | `1x10x10x384` | 384 | no: padding/stride or layout |
| 0 | 88 | DWCONV | `1x5x5x512` | `1x3x3x512` | `1x5x5x512` | 512 | no: padding/stride or layout |
| 0 | 89 | DWCONV | `1x5x5x512` | `1x3x3x512` | `1x5x5x512` | 512 | no: padding/stride or layout |
| 0 | 90 | DWCONV | `1x3x3x256` | `1x3x3x256` | `1x3x3x256` | 256 | no: padding/stride or layout |
| 0 | 91 | DWCONV | `1x3x3x256` | `1x3x3x256` | `1x3x3x256` | 256 | no: padding/stride or layout |
| 0 | 92 | DWCONV | `1x2x2x256` | `1x3x3x256` | `1x2x2x256` | 256 | no: padding/stride or layout |
| 0 | 93 | DWCONV | `1x2x2x256` | `1x3x3x256` | `1x2x2x256` | 256 | no: padding/stride or layout |
| 0 | 94 | DWCONV | `1x1x1x128` | `1x3x3x128` | `1x1x1x128` | 128 | no: padding/stride or layout |
| 0 | 95 | DWCONV | `1x1x1x128` | `1x3x3x128` | `1x1x1x128` | 128 | no: padding/stride or layout |
| 0 | 96 | CONV | `1x20x20x96` | `12x1x1x96` | `0x0x0x0` | 1 | no: dynamic/zero dim |
| 0 | 97 | CONV | `1x20x20x96` | `273x1x1x96` | `1x20x20x273` | 1 | yes |
| 0 | 98 | CONV | `1x10x10x384` | `24x1x1x384` | `1x10x10x24` | 1 | yes |
| 0 | 99 | CONV | `1x10x10x384` | `546x1x1x384` | `1x10x10x546` | 1 | yes |
| 0 | 100 | CONV | `1x5x5x512` | `24x1x1x512` | `1x5x5x24` | 1 | yes |
| 0 | 101 | CONV | `1x5x5x512` | `546x1x1x512` | `1x5x5x546` | 1 | yes |
| 0 | 102 | CONV | `1x3x3x256` | `24x1x1x256` | `1x3x3x24` | 1 | yes |
| 0 | 103 | CONV | `1x3x3x256` | `546x1x1x256` | `1x3x3x546` | 1 | yes |
| 0 | 104 | CONV | `1x2x2x256` | `24x1x1x256` | `1x2x2x24` | 1 | yes |
| 0 | 105 | CONV | `1x2x2x256` | `546x1x1x256` | `1x2x2x546` | 1 | yes |
| 0 | 106 | CONV | `1x1x1x128` | `24x1x1x128` | `1x1x1x24` | 1 | yes |
| 0 | 107 | CONV | `1x1x1x128` | `546x1x1x128` | `1x1x1x546` | 1 | yes |

### Comparison Result

Direct answers to the current coverage questions:

- `conv.py` test cases on `conv_mesa.py`: not proven as a complete pass set. `conv_mesa.py` now passes the checked default, grouped-validation, and padded/strided synthetic NPU cases, but the full `experimental/mainline6_18/conv.py` regression list has not been exhaustively replayed through `conv_mesa.py`.
- Real Mesa on all `conv.py` shapes: not proven, and the extracted evidence does not show that. Mesa/Teflon was validated on the real model examples listed above; `conv.py` also contains many standalone FP16 regression shapes that do not appear in those Teflon model graphs.
- `experimental/mainline6_18/conv.py` on all extracted Mesa shapes: no. It only matches the extracted rows marked `yes` in `conv.py exact`. The 112 extracted `no` rows need padding, stride, dynamic/zero-dim handling, quantized semantics, or depthwise layout behavior outside the current `conv.py` contract.

- Mesa/Teflon runs quantized TFLite convolution graphs with padding, stride, depthwise layout, split delegate subgraphs, and post-processing heads. Those produce many shapes that are not exact calls to the current FP16 `experimental/mainline6_18/conv.py` API.
- Of the 304 extracted Mesa/Teflon conv rows, 192 rows are exact stride-1 valid-conv shapes and 112 rows are Mesa-only shape semantics for current `conv.py` purposes.
- After deduplication, those 192 exact Mesa rows represent 124 unique shape keys. 114 of those unique exact Mesa shapes are not currently listed in the `experimental/mainline6_18/conv.py` standalone regression list. Conversely, `conv.py` has 268 unique standalone shapes that do not appear in the extracted Mesa examples.
- `experimental/mainline6_18/conv.py` can run many FP16 standalone no-padding/no-stride shapes that do not appear in the Mesa examples. Those are `conv.py`-only tests, not Mesa example failures.
- The useful overlap is the subset marked `yes` in `conv.py exact`; those can be compared as shape-compatible standalone convolutions, while the `no` rows require adding padding/stride/quantized semantics before an exact `conv_mesa.py` vs Mesa comparison is meaningful.

### Execution Audit

Local runs completed on this machine:

| Scope | Runner | Result |
|------|--------|--------|
| `experimental/mainline6_18/conv.py` full built-in list | `python3 experimental/mainline6_18/conv.py` | `100/100 PASS` |
| Extracted Mesa exact rows, deduplicated | `conv.py` replay against [this table](#mobilenetv1) onward | `62 PASS / 62 FAIL / 0 SKIP` over `124` unique exact shapes |
| `experimental/mainline6_18/conv_mesa.py` full extracted-and-regression replay | custom local runner | partial only; stopped on large shapes after real FAILs and buffer-size errors |
| Mesa-native one-layer model generation | `build/src/gallium/targets/teflon/test_teflon generate_model ...` | works here after `-Dbuild-tests=true`; no TensorFlow needed for this path |
| Mesa-native generated model execution | `tflite_runtime` CPU vs `libteflon.so` delegate | verified on `4x4x3 -> 2x2x4` conv, `max_diff=0.0` |

### Mesa Implementation Mapping

Local Mesa source used for this comparison:

- `~/mesa/src/gallium/drivers/rocket/rkt_ml.c`: lowers Teflon convolution ops into Rocket operations, records NHWC input/output shapes, padding, stride, depthwise state, zero points, scales, and weight/bias buffers.
- `~/mesa/src/gallium/drivers/rocket/rkt_task.c`: computes CBUF entries/banks, split tasks, input/output offsets, padding per slice, weight reuse, and aligned channel counts.
- `~/mesa/src/gallium/drivers/rocket/rkt_regcmd.c`: emits CNA/CORE/DPU register command streams, including `CNA_PAD_CON0`, `CNA_CONV_CON3` stride fields, depthwise mode, CBUF bank config, DPU output surfaces, and PC chaining fields.
- `~/mesa/src/gallium/drivers/rocket/rkt_coefs.c`: packs quantized weights/biases, including depthwise-specific weight layout.

`experimental/mainline6_18/conv_mesa.py` is still an FP16 standalone bring-up script, not a full Teflon quantized compiler. It now mirrors the relevant Mesa shape semantics through a stable decomposed path: non-default `stride` or `padding` routes through exact 1x1 NPU submits over padded/cropped input windows. Direct Mesa-style spatial register programming remains opt-in behind `CONV_MESA_DIRECT=1` because the direct path is shape-sensitive.

Concrete run evidence:

- Mesa/Teflon `mobilenetv1` ran end-to-end from `~/mesa` with the command style shown above and reported `0.866667: military uniform`, `time: 11.519ms`.
- A `conv.py` standalone shape absent from the extracted Mesa examples, `conv2d_2x2_1x1_4x4`, ran on the mainline Rocket path and passed CPU comparison with `max_diff=0.0017`.
- `experimental/mainline6_18/conv_mesa.py` now has an optional padded/strided decomposition path for Mesa/TFLite-style shape semantics. A synthetic `in=1x5x5`, `out_c=6`, `kernel=3x3`, `stride=2`, `padding=1,1,1,1` case ran on the NPU and matched CPU with `max_diff=0.0000`.
- The extracted Mesa rows marked `no` are the current "Mesa can run this model op shape, `conv.py` cannot run the exact same semantics" set because they require padding, stride, dynamic tensor dimensions, or quantized/depthwise layout behavior outside `conv.py`'s current FP16 valid-conv contract.

## Known Driver Bug

Models failing with the assertion `input_op_1' failed` in `rkt_ml_subgraph_create` (`src/gallium/drivers/rocket/rkt_ml.c:343`):

- **efficientdet** — 320x320 uint8, SSD post-processing ops
- **movenetlightning** / **movenetthunder** — single-subgraph, float32 keypoint output
- **yolox** — 416x416 int8, single output tensor 3549x85

The root cause appears to be the rocket driver not handling certain graph structures — likely models where the first operation in a subgraph is not directly fed by the subgraph input tensor. All these models run correctly on CPU (XNNPACK delegate).

## Complete Shape Result Matrix

This matrix is the union of the `experimental/mainline6_18/conv.py` test list plus the extracted Mesa/Teflon conv rows in this file, deduplicated by batch/input/weight/groups/stride/padding shape key. Status values mean: `PASS` and `FAIL` are executed comparisons; `ERROR` is an attempted run that raised an exception; `SKIP` is an intentional semantic mismatch such as Mesa-only padding/stride for `conv.py`; `UNSUPPORTED` means the Mesa native `test_teflon generate_model` helper cannot represent that shape; `NOT_RUN` means the partial `conv_mesa.py` replay did not reach that shape.

Summary from local logs and generated-model replay:

| Target | PASS | FAIL | ERROR | SKIP | UNSUPPORTED | NOT_RUN | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mainline6_18/conv.py` | 359 | 46 | 1 | 47 | 0 | 0 | 453 |
| `mainline6_18/conv_mesa.py` | 65 | 49 | 320 | 10 | 5 | 4 | 453 |
| real Mesa custom one-layer TFLite | 399 | 5 | 4 | 0 | 45 | 0 | 453 |

| Shape ID | Sources | Input NCHW | Weight OIHW | Groups | mainline `conv.py` | `conv_mesa.py` | real Mesa custom TFLite |
|---|---|---|---|---:|---|---|---|
| `b1_c1_h4_w4_oc6_wic1_k1x1_g1_s1_pvalid` | `conv.py` | `1x1x4x4` | `6x1x1x1` | 1 | PASS: max_diff=0.0008 | PASS: max_diff=0.0009 sec=0.00 | PASS: PASS max_diff=0.0 out=1x4x4x6 |
| `b1_c3_h4_w4_oc3_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x4x4` | `3x3x1x1` | 1 | PASS: max_diff=0.0018 | PASS: max_diff=0.0009 sec=0.00 | PASS: PASS max_diff=0.0 out=1x4x4x3 |
| `b1_c4_h4_w4_oc2_wic4_k1x1_g1_s1_pvalid` | `conv.py` | `1x4x4x4` | `2x4x1x1` | 1 | PASS: max_diff=0.0010 | PASS: max_diff=0.0009 sec=0.00 | PASS: PASS max_diff=0.0 out=1x4x4x2 |
| `b1_c4_h9_w9_oc4_wic4_k1x1_g1_s1_pvalid` | `conv.py` | `1x4x9x9` | `4x4x1x1` | 1 | PASS: max_diff=0.0039 | PASS: max_diff=0.0014 sec=0.00 | PASS: PASS max_diff=0.0 out=1x9x9x4 |
| `b1_c16_h8_w8_oc16_wic16_k1x1_g1_s1_pvalid` | `conv.py` | `1x16x8x8` | `16x16x1x1` | 1 | PASS: max_diff=0.0041 | PASS: max_diff=0.0070 sec=0.00 | PASS: PASS max_diff=0.0 out=1x8x8x16 |
| `b1_c16_h32_w32_oc16_wic16_k1x1_g1_s1_pvalid` | `conv.py` | `1x16x32x32` | `16x16x1x1` | 1 | PASS: max_diff=0.0077 | PASS: max_diff=0.0100 sec=0.01 | PASS: PASS max_diff=0.0 out=1x32x32x16 |
| `b1_c4_h9_w9_oc4_wic4_k3x3_g1_s1_pvalid` | `conv.py` | `1x4x9x9` | `4x4x3x3` | 1 | PASS: max_diff=0.0074 | PASS: max_diff=0.0164 sec=0.02 | PASS: PASS max_diff=0.0 out=1x7x7x4 |
| `b1_c16_h18_w18_oc16_wic16_k3x3_g1_s1_pvalid` | `conv.py` | `1x16x18x18` | `16x16x3x3` | 1 | PASS: max_diff=0.0154 | FAIL: max_diff=0.1559 sec=0.12 | PASS: PASS max_diff=0.0 out=1x16x16x16 |
| `b2_c4_h9_w9_oc4_wic4_k3x3_g1_s1_pvalid` | `conv.py` | `2x4x9x9` | `4x4x3x3` | 1 | PASS: max_diff=0.0077 | SKIP: unsupported_batch=2 | UNSUPPORTED: batch |
| `b1_c1_h5_w7_oc6_wic1_k3x3_g1_s1_pvalid` | `conv.py` | `1x1x5x7` | `6x1x3x3` | 1 | PASS: max_diff=0.0030 | PASS: max_diff=0.0063 sec=0.01 | UNSUPPORTED: non_square_input |
| `b1_c3_h11_w28_oc3_wic1_k3x3_g3_s1_pvalid` | `conv.py` | `1x3x11x28` | `3x1x3x3` | 3 | PASS: max_diff=0.0038 | PASS: max_diff=0.0075 sec=0.02 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w5_oc6_wic3_k1x3_g1_s1_pvalid` | `conv.py` | `1x3x5x5` | `6x3x1x3` | 1 | PASS: max_diff=0.0039 | PASS: max_diff=0.0056 sec=0.01 | UNSUPPORTED: non_square_kernel |
| `b1_c3_h5_w7_oc6_wic1_k3x3_g3_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x1x3x3` | 3 | PASS: max_diff=0.0023 | PASS: max_diff=0.0040 sec=0.02 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k2x1_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x2x1` | 1 | PASS: max_diff=0.0036 | PASS: max_diff=0.0043 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k2x3_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x2x3` | 1 | PASS: max_diff=0.0038 | PASS: max_diff=0.0125 sec=0.01 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k2x5_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x2x5` | 1 | PASS: max_diff=0.0053 | PASS: max_diff=0.0169 sec=0.02 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k3x1_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x3x1` | 1 | PASS: max_diff=0.0025 | PASS: max_diff=0.0048 sec=0.01 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k3x3_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x3x3` | 1 | PASS: max_diff=0.0057 | PASS: max_diff=0.0120 sec=0.02 | UNSUPPORTED: non_square_input |
| `b1_c3_h5_w7_oc6_wic3_k3x5_g1_s1_pvalid` | `conv.py` | `1x3x5x7` | `6x3x3x5` | 1 | PASS: max_diff=0.0045 | PASS: max_diff=0.0275 sec=0.03 | UNSUPPORTED: non_square_input |
| `b1_c1_h5_w7_oc6_wic1_k2x1_g1_s1_pvalid` | `conv.py` | `1x1x5x7` | `6x1x2x1` | 1 | PASS: max_diff=0.0009 | PASS: max_diff=0.0015 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c1_h5_w7_oc6_wic1_k2x3_g1_s1_pvalid` | `conv.py` | `1x1x5x7` | `6x1x2x3` | 1 | PASS: max_diff=0.0038 | PASS: max_diff=0.0043 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c1_h5_w7_oc6_wic1_k3x1_g1_s1_pvalid` | `conv.py` | `1x1x5x7` | `6x1x3x1` | 1 | PASS: max_diff=0.0017 | PASS: max_diff=0.0030 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c1_h5_w7_oc6_wic1_k3x5_g1_s1_pvalid` | `conv.py` | `1x1x5x7` | `6x1x3x5` | 1 | PASS: max_diff=0.0036 | PASS: max_diff=0.0137 sec=0.01 | UNSUPPORTED: non_square_input |
| `b1_c4_h1_w1_oc2_wic2_k1x1_g2_s1_pvalid` | `conv.py` | `1x4x1x1` | `2x2x1x1` | 2 | PASS: max_diff=0.0004 | PASS: max_diff=0.0004 sec=0.00 | UNSUPPORTED: grouped |
| `b1_c4_h1_w1_oc4_wic2_k1x1_g2_s1_pvalid` | `conv.py` | `1x4x1x1` | `4x2x1x1` | 2 | PASS: max_diff=0.0006 | PASS: max_diff=0.0002 sec=0.00 | UNSUPPORTED: grouped |
| `b1_c32_h32_w32_oc32_wic1_k1x1_g32_s1_pvalid` | `conv.py` | `1x32x32x32` | `32x1x1x1` | 32 | PASS: max_diff=0.0010 | PASS: max_diff=0.0019 sec=0.01 | PASS: PASS max_diff=0.0 out=1x32x32x32 |
| `b1_c15_h5_w5_oc35_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `1x15x5x5` | `35x3x3x3` | 5 | PASS: max_diff=0.0072 | PASS: max_diff=0.0194 sec=0.09 | UNSUPPORTED: grouped |
| `b2_c3_h11_w28_oc3_wic1_k3x3_g3_s1_pvalid` | `conv.py` | `2x3x11x28` | `3x1x3x3` | 3 | PASS: max_diff=0.0038 | SKIP: unsupported_batch=2 | UNSUPPORTED: batch |
| `b4_c15_h5_w5_oc35_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `4x15x5x5` | `35x3x3x3` | 5 | PASS: max_diff=0.0075 | SKIP: unsupported_batch=4 | UNSUPPORTED: batch |
| `b1_c4_h5_w5_oc4_wic2_k3x3_g2_s1_pvalid` | `conv.py` | `1x4x5x5` | `4x2x3x3` | 2 | PASS: max_diff=0.0032 | PASS: max_diff=0.0053 sec=0.02 | UNSUPPORTED: grouped |
| `b1_c4_h5_w5_oc8_wic2_k3x3_g2_s1_pvalid` | `conv.py` | `1x4x5x5` | `8x2x3x3` | 2 | PASS: max_diff=0.0066 | PASS: max_diff=0.0104 sec=0.02 | UNSUPPORTED: grouped |
| `b1_c4_h5_w5_oc12_wic2_k3x3_g2_s1_pvalid` | `conv.py` | `1x4x5x5` | `12x2x3x3` | 2 | PASS: max_diff=0.0066 | PASS: max_diff=0.0104 sec=0.02 | UNSUPPORTED: grouped |
| `b1_c6_h5_w5_oc6_wic2_k3x3_g3_s1_pvalid` | `conv.py` | `1x6x5x5` | `6x2x3x3` | 3 | PASS: max_diff=0.0038 | PASS: max_diff=0.0059 sec=0.03 | UNSUPPORTED: grouped |
| `b1_c6_h5_w5_oc12_wic2_k3x3_g3_s1_pvalid` | `conv.py` | `1x6x5x5` | `12x2x3x3` | 3 | PASS: max_diff=0.0039 | PASS: max_diff=0.0050 sec=0.03 | UNSUPPORTED: grouped |
| `b1_c6_h5_w5_oc18_wic2_k3x3_g3_s1_pvalid` | `conv.py` | `1x6x5x5` | `18x2x3x3` | 3 | PASS: max_diff=0.0039 | PASS: max_diff=0.0119 sec=0.04 | UNSUPPORTED: grouped |
| `b1_c15_h5_w5_oc20_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `1x15x5x5` | `20x3x3x3` | 5 | PASS: max_diff=0.0055 | PASS: max_diff=0.0123 sec=0.08 | UNSUPPORTED: grouped |
| `b1_c15_h5_w5_oc25_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `1x15x5x5` | `25x3x3x3` | 5 | PASS: max_diff=0.0043 | PASS: max_diff=0.0188 sec=0.09 | UNSUPPORTED: grouped |
| `b1_c15_h5_w5_oc30_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `1x15x5x5` | `30x3x3x3` | 5 | PASS: max_diff=0.0066 | PASS: max_diff=0.0188 sec=0.17 | UNSUPPORTED: grouped |
| `b1_c15_h5_w5_oc40_wic3_k3x3_g5_s1_pvalid` | `conv.py` | `1x15x5x5` | `40x3x3x3` | 5 | PASS: max_diff=0.0072 | PASS: max_diff=0.0194 sec=0.16 | UNSUPPORTED: grouped |
| `b1_c2_h4_w4_oc2_wic2_k1x1_g1_s1_pvalid` | `conv.py` | `1x2x4x4` | `2x2x1x1` | 1 | PASS: max_diff=0.0017 | PASS: max_diff=0.0006 sec=0.00 | PASS: PASS max_diff=0.0 out=1x4x4x2 |
| `b1_c8_h5_w5_oc8_wic8_k1x1_g1_s1_pvalid` | `conv.py` | `1x8x5x5` | `8x8x1x1` | 1 | PASS: max_diff=0.0035 | PASS: max_diff=0.0044 sec=0.00 | PASS: PASS max_diff=0.0 out=1x5x5x8 |
| `b1_c10_h9_w9_oc20_wic10_k3x3_g1_s1_pvalid` | `conv.py` | `1x10x9x9` | `20x10x3x3` | 1 | PASS: max_diff=0.0091 | PASS: max_diff=0.0648 sec=0.12 | PASS: PASS max_diff=0.0 out=1x7x7x20 |
| `b1_c16_h9_w9_oc16_wic16_k3x3_g1_s1_pvalid` | `conv.py` | `1x16x9x9` | `16x16x3x3` | 1 | PASS: max_diff=0.0155 | FAIL: max_diff=0.1209 sec=0.10 | PASS: PASS max_diff=0.0 out=1x7x7x16 |
| `b1_c2_h6_w6_oc4_wic2_k3x3_g1_s1_pvalid` | `conv.py` | `1x2x6x6` | `4x2x3x3` | 1 | PASS: max_diff=0.0035 | PASS: max_diff=0.0118 sec=0.01 | PASS: PASS max_diff=0.0 out=1x4x4x4 |
| `b1_c2_h5_w5_oc4_wic2_k2x2_g1_s1_pvalid` | `conv.py` | `1x2x5x5` | `4x2x2x2` | 1 | PASS: max_diff=0.0028 | PASS: max_diff=0.0033 sec=0.01 | PASS: PASS max_diff=0.0 out=1x4x4x4 |
| `b1_c1_h10_w10_oc32_wic1_k5x5_g1_s1_pvalid` | `conv.py` | `1x1x10x10` | `32x1x5x5` | 1 | PASS: max_diff=0.0075 | PASS: max_diff=0.0227 sec=0.04 | PASS: PASS max_diff=0.0 out=1x6x6x32 |
| `b1_c8_h10_w10_oc4_wic8_k4x4_g1_s1_pvalid` | `conv.py` | `1x8x10x10` | `4x8x4x4` | 1 | PASS: max_diff=0.0132 | PASS: max_diff=0.0912 sec=0.08 | PASS: PASS max_diff=0.0 out=1x7x7x4 |
| `b1_c3_h224_w224_oc32_wic3_k3x3_g1_s1_pvalid` | `conv.py+mesa.md` | `1x3x224x224` | `32x3x3x3` | 1 | PASS: max_diff=0.0151 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x222x222x32 |
| `b1_c32_h112_w112_oc32_wic1_k3x3_g32_s1_pvalid` | `conv.py+mesa.md` | `1x32x112x112` | `32x1x3x3` | 32 | PASS: max_diff=0.0075 | FAIL: max_diff=30.8104 sec=0.05 | PASS: PASS max_diff=1.0 out=1x110x110x32 |
| `b1_c32_h112_w112_oc64_wic32_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x32x112x112` | `64x32x1x1` | 1 | PASS: max_diff=0.0146 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x112x112x64 |
| `b1_c64_h112_w112_oc64_wic1_k3x3_g64_s1_pvalid` | `conv.py+mesa.md` | `1x64x112x112` | `64x1x3x3` | 64 | PASS: max_diff=0.0075 | FAIL: max_diff=29.5156 sec=0.10 | PASS: PASS max_diff=1.0 out=1x110x110x64 |
| `b1_c64_h56_w56_oc128_wic64_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x64x56x56` | `128x64x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: ValueError: buffer is smaller than requested size sec=0.53 | PASS: PASS max_diff=1.0 out=1x56x56x128 |
| `b1_c128_h56_w56_oc128_wic1_k3x3_g128_s1_pvalid` | `conv.py+mesa.md` | `1x128x56x56` | `128x1x3x3` | 128 | PASS: max_diff=0.0078 | FAIL: max_diff=25.5057 sec=0.06 | PASS: PASS max_diff=1.0 out=1x54x54x128 |
| `b1_c128_h56_w56_oc128_wic128_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x128x56x56` | `128x128x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x56x56x128 |
| `b1_c128_h28_w28_oc256_wic128_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x128x28x28` | `256x128x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=0.0 out=1x28x28x256 |
| `b1_c256_h28_w28_oc256_wic1_k3x3_g256_s1_pvalid` | `conv.py+mesa.md` | `1x256x28x28` | `256x1x3x3` | 256 | PASS: max_diff=0.0076 | FAIL: max_diff=26.0331 sec=0.06 | PASS: PASS max_diff=1.0 out=1x26x26x256 |
| `b1_c256_h28_w28_oc256_wic256_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x256x28x28` | `256x256x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x28x28x256 |
| `b1_c256_h14_w14_oc512_wic256_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x256x14x14` | `512x256x1x1` | 1 | PASS: max_diff=0.0311 | FAIL: max_diff=809.2072 sec=35.25 | PASS: PASS max_diff=0.0 out=1x14x14x512 |
| `b1_c512_h14_w14_oc512_wic1_k3x3_g512_s1_pvalid` | `conv.py+mesa.md` | `1x512x14x14` | `512x1x3x3` | 512 | PASS: max_diff=0.0076 | FAIL: max_diff=27.7646 sec=0.08 | PASS: PASS max_diff=1.0 out=1x12x12x512 |
| `b1_c512_h14_w14_oc512_wic512_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x512x14x14` | `512x512x1x1` | 1 | PASS: max_diff=0.0312 | FAIL: max_diff=1659.2895 sec=70.71 | PASS: PASS max_diff=1.0 out=1x14x14x512 |
| `b1_c512_h7_w7_oc1024_wic512_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x512x7x7` | `1024x512x1x1` | 1 | PASS: max_diff=0.0312 | FAIL: max_diff=1673.8350 sec=70.74 | PASS: PASS max_diff=0.0 out=1x7x7x1024 |
| `b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024_s1_pvalid` | `conv.py+mesa.md` | `1x1024x7x7` | `1024x1x3x3` | 1024 | PASS: max_diff=0.0064 | FAIL: max_diff=20.8329 sec=0.15 | PASS: PASS max_diff=1.0 out=1x5x5x1024 |
| `b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x1024x7x7` | `1024x1024x1x1` | 1 | PASS: max_diff=0.0606 | FAIL: max_diff=3446.8567 sec=142.19 | PASS: PASS max_diff=1.0 out=1x7x7x1024 |
| `b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024_s1_pvalid` | `conv.py` | `1x1024x7x7` | `1024x1x7x7` | 1024 | PASS: max_diff=0.0078 | FAIL: max_diff=32.1104 sec=0.52 | PASS: PASS max_diff=0.0 out=1x1x1x1024 |
| `b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1_s1_pvalid` | `conv.py+mesa.md` | `1x1024x1x1` | `1001x1024x1x1` | 1 | PASS: max_diff=0.0314 | FAIL: max_diff=2252.5672 sec=9.88 | PASS: PASS max_diff=0.0 out=1x1x1x1001 |
| `b1_c1_h1_w11_oc6_wic1_k1x1_g1_s1_pvalid` | `conv.py` | `1x1x1x11` | `6x1x1x1` | 1 | PASS: max_diff=0.0009 | PASS: max_diff=0.0007 sec=0.00 | UNSUPPORTED: non_square_input |
| `b8_c1_h1_w11_oc6_wic1_k1x1_g1_s1_pvalid` | `conv.py` | `8x1x1x11` | `6x1x1x1` | 1 | PASS: max_diff=0.0010 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b1_c1_h1_w11_oc6_wic1_k1x2_g1_s1_pvalid` | `conv.py` | `1x1x1x11` | `6x1x1x2` | 1 | PASS: max_diff=0.0018 | PASS: max_diff=0.0010 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c1_h1_w11_oc6_wic1_k1x5_g1_s1_pvalid` | `conv.py` | `1x1x1x11` | `6x1x1x5` | 1 | PASS: max_diff=0.0037 | PASS: max_diff=0.0023 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c3_h1_w11_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x1x11` | `6x3x1x1` | 1 | PASS: max_diff=0.0036 | PASS: max_diff=0.0015 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c3_h1_w11_oc6_wic3_k1x2_g1_s1_pvalid` | `conv.py` | `1x3x1x11` | `6x3x1x2` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0046 sec=0.00 | UNSUPPORTED: non_square_input |
| `b1_c3_h1_w11_oc6_wic3_k1x5_g1_s1_pvalid` | `conv.py` | `1x3x1x11` | `6x3x1x5` | 1 | PASS: max_diff=0.0037 | PASS: max_diff=0.0086 sec=0.02 | UNSUPPORTED: non_square_input |
| `b1_c3_h1_w11_oc6_wic1_k1x5_g3_s1_pvalid` | `conv.py` | `1x3x1x11` | `6x1x1x5` | 3 | PASS: max_diff=0.0021 | PASS: max_diff=0.0024 sec=0.02 | UNSUPPORTED: non_square_input |
| `b8_c1_h1_w11_oc6_wic1_k1x2_g1_s1_pvalid` | `conv.py` | `8x1x1x11` | `6x1x1x2` | 1 | PASS: max_diff=0.0019 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b8_c1_h1_w11_oc6_wic1_k1x5_g1_s1_pvalid` | `conv.py` | `8x1x1x11` | `6x1x1x5` | 1 | PASS: max_diff=0.0031 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b8_c3_h1_w11_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `8x3x1x11` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b8_c3_h1_w11_oc6_wic3_k1x2_g1_s1_pvalid` | `conv.py` | `8x3x1x11` | `6x3x1x2` | 1 | PASS: max_diff=0.0029 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b8_c3_h1_w11_oc6_wic3_k1x5_g1_s1_pvalid` | `conv.py` | `8x3x1x11` | `6x3x1x5` | 1 | PASS: max_diff=0.0039 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b8_c3_h1_w11_oc6_wic1_k1x5_g3_s1_pvalid` | `conv.py` | `8x3x1x11` | `6x1x1x5` | 3 | PASS: max_diff=0.0020 | SKIP: unsupported_batch=8 | UNSUPPORTED: batch |
| `b1_c3_h2_w2_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x2x2` | `6x3x1x1` | 1 | PASS: max_diff=0.0009 | PASS: max_diff=0.0009 sec=0.00 | PASS: PASS max_diff=0.0 out=1x2x2x6 |
| `b1_c3_h4_w4_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x4x4` | `6x3x1x1` | 1 | PASS: max_diff=0.0018 | PASS: max_diff=0.0010 sec=0.00 | PASS: PASS max_diff=0.0 out=1x4x4x6 |
| `b1_c3_h6_w6_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x6x6` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0019 sec=0.00 | PASS: PASS max_diff=0.0 out=1x6x6x6 |
| `b1_c3_h8_w8_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x8x8` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0035 sec=0.00 | PASS: PASS max_diff=0.0 out=1x8x8x6 |
| `b1_c3_h10_w10_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x10x10` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | PASS: max_diff=0.0010 sec=0.00 | PASS: PASS max_diff=0.0 out=1x10x10x6 |
| `b1_c3_h12_w12_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x12x12` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | PASS: max_diff=0.0019 sec=0.00 | PASS: PASS max_diff=0.0 out=1x12x12x6 |
| `b1_c3_h14_w14_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x14x14` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0018 sec=0.00 | PASS: PASS max_diff=0.0 out=1x14x14x6 |
| `b1_c3_h16_w16_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x16x16` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | PASS: max_diff=0.0018 sec=0.00 | PASS: PASS max_diff=0.0 out=1x16x16x6 |
| `b1_c3_h18_w18_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x18x18` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | PASS: max_diff=0.0035 sec=0.00 | PASS: PASS max_diff=0.0 out=1x18x18x6 |
| `b1_c3_h20_w20_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x20x20` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | PASS: max_diff=0.0018 sec=0.00 | PASS: PASS max_diff=0.0 out=1x20x20x6 |
| `b1_c3_h22_w22_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x22x22` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | PASS: max_diff=0.0019 sec=0.00 | PASS: PASS max_diff=0.0 out=1x22x22x6 |
| `b1_c3_h24_w24_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x24x24` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0019 sec=0.00 | PASS: PASS max_diff=0.0 out=1x24x24x6 |
| `b1_c3_h26_w26_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x26x26` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | PASS: max_diff=0.0035 sec=0.00 | PASS: PASS max_diff=0.0 out=1x26x26x6 |
| `b1_c3_h28_w28_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x28x28` | `6x3x1x1` | 1 | PASS: max_diff=0.0028 | PASS: max_diff=0.0020 sec=0.00 | PASS: PASS max_diff=0.0 out=1x28x28x6 |
| `b1_c3_h30_w30_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x30x30` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | PASS: max_diff=0.0019 sec=0.00 | PASS: PASS max_diff=0.0 out=1x30x30x6 |
| `b1_c3_h32_w32_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x32x32` | `6x3x1x1` | 1 | PASS: max_diff=0.0030 | PASS: max_diff=0.0033 sec=0.00 | PASS: PASS max_diff=0.0 out=1x32x32x6 |
| `b1_c3_h34_w34_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x34x34` | `6x3x1x1` | 1 | PASS: max_diff=0.0024 | FAIL: max_diff=13.2914 sec=0.00 | PASS: PASS max_diff=0.0 out=1x34x34x6 |
| `b1_c3_h36_w36_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x36x36` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | FAIL: max_diff=8.3905 sec=0.00 | PASS: PASS max_diff=0.0 out=1x36x36x6 |
| `b1_c3_h38_w38_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x38x38` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | FAIL: max_diff=7.2191 sec=0.00 | PASS: PASS max_diff=0.0 out=1x38x38x6 |
| `b1_c3_h40_w40_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x40x40` | `6x3x1x1` | 1 | PASS: max_diff=0.0033 | FAIL: max_diff=7.1156 sec=0.00 | PASS: PASS max_diff=0.0 out=1x40x40x6 |
| `b1_c3_h42_w42_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x42x42` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | FAIL: max_diff=10.1539 sec=0.00 | PASS: PASS max_diff=0.0 out=1x42x42x6 |
| `b1_c3_h44_w44_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x44x44` | `6x3x1x1` | 1 | PASS: max_diff=0.0035 | FAIL: max_diff=15.5540 sec=0.00 | PASS: PASS max_diff=0.0 out=1x44x44x6 |
| `b1_c3_h46_w46_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x46x46` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | FAIL: max_diff=13.8537 sec=0.00 | PASS: PASS max_diff=0.0 out=1x46x46x6 |
| `b1_c3_h48_w48_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x48x48` | `6x3x1x1` | 1 | PASS: max_diff=0.0019 | FAIL: max_diff=12.3080 sec=0.00 | PASS: PASS max_diff=0.0 out=1x48x48x6 |
| `b1_c3_h50_w50_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x50x50` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | FAIL: max_diff=12.4873 sec=0.00 | PASS: PASS max_diff=0.0 out=1x50x50x6 |
| `b1_c3_h52_w52_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x52x52` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=0.0 out=1x52x52x6 |
| `b1_c3_h54_w54_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x54x54` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=0.0 out=1x54x54x6 |
| `b1_c3_h56_w56_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x56x56` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x56x56x6 |
| `b1_c3_h58_w58_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x58x58` | `6x3x1x1` | 1 | PASS: max_diff=0.0023 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x58x58x6 |
| `b1_c3_h60_w60_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x60x60` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x60x60x6 |
| `b1_c3_h62_w62_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x62x62` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x62x62x6 |
| `b1_c3_h64_w64_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x64x64` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x64x64x6 |
| `b1_c3_h66_w66_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x66x66` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x66x66x6 |
| `b1_c3_h68_w68_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x68x68` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x68x68x6 |
| `b1_c3_h70_w70_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x70x70` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x70x70x6 |
| `b1_c3_h72_w72_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x72x72` | `6x3x1x1` | 1 | PASS: max_diff=0.0031 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x72x72x6 |
| `b1_c3_h74_w74_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x74x74` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x74x74x6 |
| `b1_c3_h76_w76_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x76x76` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x76x76x6 |
| `b1_c3_h78_w78_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x78x78` | `6x3x1x1` | 1 | PASS: max_diff=0.0021 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x78x78x6 |
| `b1_c3_h80_w80_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x80x80` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x80x80x6 |
| `b1_c3_h82_w82_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x82x82` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x82x82x6 |
| `b1_c3_h84_w84_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x84x84` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x84x84x6 |
| `b1_c3_h86_w86_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x86x86` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x86x86x6 |
| `b1_c3_h88_w88_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x88x88` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x88x88x6 |
| `b1_c3_h90_w90_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x90x90` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x90x90x6 |
| `b1_c3_h92_w92_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x92x92` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x92x92x6 |
| `b1_c3_h94_w94_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x94x94` | `6x3x1x1` | 1 | PASS: max_diff=0.0021 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x94x94x6 |
| `b1_c3_h96_w96_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x96x96` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x96x96x6 |
| `b1_c3_h98_w98_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x98x98` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x98x98x6 |
| `b1_c3_h100_w100_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x100x100` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x100x100x6 |
| `b1_c3_h102_w102_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x102x102` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x102x102x6 |
| `b1_c3_h104_w104_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x104x104` | `6x3x1x1` | 1 | PASS: max_diff=0.0033 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x104x104x6 |
| `b1_c3_h106_w106_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x106x106` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x106x106x6 |
| `b1_c3_h108_w108_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x108x108` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x108x108x6 |
| `b1_c3_h110_w110_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x110x110` | `6x3x1x1` | 1 | PASS: max_diff=0.0028 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x110x110x6 |
| `b1_c3_h112_w112_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x112x112` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x112x112x6 |
| `b1_c3_h114_w114_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x114x114` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x114x114x6 |
| `b1_c3_h116_w116_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x116x116` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x116x116x6 |
| `b1_c3_h118_w118_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x118x118` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x118x118x6 |
| `b1_c3_h120_w120_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x120x120` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x120x120x6 |
| `b1_c3_h122_w122_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x122x122` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x122x122x6 |
| `b1_c3_h124_w124_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x124x124` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x124x124x6 |
| `b1_c3_h126_w126_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x126x126` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x126x126x6 |
| `b1_c3_h128_w128_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x128x128` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x128x128x6 |
| `b1_c3_h130_w130_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x130x130` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x130x130x6 |
| `b1_c3_h132_w132_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x132x132` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x132x132x6 |
| `b1_c3_h134_w134_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x134x134` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x134x134x6 |
| `b1_c3_h136_w136_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x136x136` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x136x136x6 |
| `b1_c3_h138_w138_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x138x138` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x138x138x6 |
| `b1_c3_h140_w140_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x140x140` | `6x3x1x1` | 1 | PASS: max_diff=0.0033 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x140x140x6 |
| `b1_c3_h142_w142_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x142x142` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x142x142x6 |
| `b1_c3_h144_w144_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x144x144` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x144x144x6 |
| `b1_c3_h146_w146_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x146x146` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x146x146x6 |
| `b1_c3_h148_w148_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x148x148` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x148x148x6 |
| `b1_c3_h150_w150_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x150x150` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x150x150x6 |
| `b1_c3_h152_w152_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x152x152` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x152x152x6 |
| `b1_c3_h154_w154_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x154x154` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x154x154x6 |
| `b1_c3_h156_w156_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x156x156` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x156x156x6 |
| `b1_c3_h158_w158_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x158x158` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.00 | PASS: PASS max_diff=1.0 out=1x158x158x6 |
| `b1_c3_h160_w160_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x160x160` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x160x160x6 |
| `b1_c3_h162_w162_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x162x162` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x162x162x6 |
| `b1_c3_h164_w164_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x164x164` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x164x164x6 |
| `b1_c3_h166_w166_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x166x166` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x166x166x6 |
| `b1_c3_h168_w168_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x168x168` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x168x168x6 |
| `b1_c3_h170_w170_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x170x170` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x170x170x6 |
| `b1_c3_h172_w172_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x172x172` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x172x172x6 |
| `b1_c3_h174_w174_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x174x174` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x174x174x6 |
| `b1_c3_h176_w176_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x176x176` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x176x176x6 |
| `b1_c3_h178_w178_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x178x178` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x178x178x6 |
| `b1_c3_h180_w180_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x180x180` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x180x180x6 |
| `b1_c3_h182_w182_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x182x182` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x182x182x6 |
| `b1_c3_h184_w184_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x184x184` | `6x3x1x1` | 1 | PASS: max_diff=0.0031 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x184x184x6 |
| `b1_c3_h186_w186_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x186x186` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x186x186x6 |
| `b1_c3_h188_w188_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x188x188` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x188x188x6 |
| `b1_c3_h190_w190_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x190x190` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x190x190x6 |
| `b1_c3_h192_w192_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x192x192` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x192x192x6 |
| `b1_c3_h194_w194_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x194x194` | `6x3x1x1` | 1 | PASS: max_diff=0.0036 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x194x194x6 |
| `b1_c3_h196_w196_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x196x196` | `6x3x1x1` | 1 | PASS: max_diff=0.0031 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x196x196x6 |
| `b1_c3_h198_w198_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x198x198` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x198x198x6 |
| `b1_c3_h200_w200_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x200x200` | `6x3x1x1` | 1 | PASS: max_diff=0.0034 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x200x200x6 |
| `b1_c3_h202_w202_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x202x202` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x202x202x6 |
| `b1_c3_h204_w204_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x204x204` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x204x204x6 |
| `b1_c3_h206_w206_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x206x206` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x206x206x6 |
| `b1_c3_h208_w208_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x208x208` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x208x208x6 |
| `b1_c3_h210_w210_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x210x210` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x210x210x6 |
| `b1_c3_h212_w212_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x212x212` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x212x212x6 |
| `b1_c3_h214_w214_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x214x214` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x214x214x6 |
| `b1_c3_h216_w216_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x216x216` | `6x3x1x1` | 1 | PASS: max_diff=0.0023 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x216x216x6 |
| `b1_c3_h218_w218_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x218x218` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x218x218x6 |
| `b1_c3_h220_w220_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x220x220` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x220x220x6 |
| `b1_c3_h222_w222_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x222x222` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x222x222x6 |
| `b1_c3_h224_w224_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x224x224` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x224x224x6 |
| `b1_c3_h226_w226_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x226x226` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x226x226x6 |
| `b1_c3_h228_w228_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x228x228` | `6x3x1x1` | 1 | PASS: max_diff=0.0033 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x228x228x6 |
| `b1_c3_h230_w230_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x230x230` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x230x230x6 |
| `b1_c3_h232_w232_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x232x232` | `6x3x1x1` | 1 | PASS: max_diff=0.0033 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x232x232x6 |
| `b1_c3_h234_w234_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x234x234` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x234x234x6 |
| `b1_c3_h236_w236_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x236x236` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x236x236x6 |
| `b1_c3_h238_w238_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x238x238` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x238x238x6 |
| `b1_c3_h240_w240_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x240x240` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x240x240x6 |
| `b1_c3_h242_w242_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x242x242` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x242x242x6 |
| `b1_c3_h244_w244_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x244x244` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x244x244x6 |
| `b1_c3_h246_w246_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x246x246` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x246x246x6 |
| `b1_c3_h248_w248_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x248x248` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x248x248x6 |
| `b1_c3_h250_w250_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x250x250` | `6x3x1x1` | 1 | PASS: max_diff=0.0031 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x250x250x6 |
| `b1_c3_h252_w252_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x252x252` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x252x252x6 |
| `b1_c3_h254_w254_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x254x254` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x254x254x6 |
| `b1_c3_h256_w256_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x256x256` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x256x256x6 |
| `b1_c3_h258_w258_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x258x258` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x258x258x6 |
| `b1_c3_h260_w260_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x260x260` | `6x3x1x1` | 1 | PASS: max_diff=0.0036 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x260x260x6 |
| `b1_c3_h262_w262_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x262x262` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x262x262x6 |
| `b1_c3_h264_w264_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x264x264` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x264x264x6 |
| `b1_c3_h266_w266_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x266x266` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x266x266x6 |
| `b1_c3_h268_w268_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x268x268` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x268x268x6 |
| `b1_c3_h270_w270_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x270x270` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x270x270x6 |
| `b1_c3_h272_w272_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x272x272` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x272x272x6 |
| `b1_c3_h274_w274_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x274x274` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x274x274x6 |
| `b1_c3_h276_w276_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x276x276` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x276x276x6 |
| `b1_c3_h278_w278_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x278x278` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x278x278x6 |
| `b1_c3_h280_w280_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x280x280` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x280x280x6 |
| `b1_c3_h282_w282_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x282x282` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x282x282x6 |
| `b1_c3_h284_w284_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x284x284` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x284x284x6 |
| `b1_c3_h286_w286_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x286x286` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x286x286x6 |
| `b1_c3_h288_w288_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x288x288` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x288x288x6 |
| `b1_c3_h290_w290_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x290x290` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x290x290x6 |
| `b1_c3_h292_w292_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x292x292` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x292x292x6 |
| `b1_c3_h294_w294_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x294x294` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x294x294x6 |
| `b1_c3_h296_w296_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x296x296` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.01 | PASS: PASS max_diff=1.0 out=1x296x296x6 |
| `b1_c3_h298_w298_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x298x298` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x298x298x6 |
| `b1_c3_h300_w300_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x300x300` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x300x300x6 |
| `b1_c3_h302_w302_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x302x302` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x302x302x6 |
| `b1_c3_h304_w304_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x304x304` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x304x304x6 |
| `b1_c3_h306_w306_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x306x306` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x306x306x6 |
| `b1_c3_h308_w308_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x308x308` | `6x3x1x1` | 1 | PASS: max_diff=0.0030 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x308x308x6 |
| `b1_c3_h310_w310_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x310x310` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x310x310x6 |
| `b1_c3_h312_w312_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x312x312` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x312x312x6 |
| `b1_c3_h314_w314_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x314x314` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x314x314x6 |
| `b1_c3_h316_w316_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x316x316` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x316x316x6 |
| `b1_c3_h318_w318_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x318x318` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x318x318x6 |
| `b1_c3_h320_w320_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x320x320` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x320x320x6 |
| `b1_c3_h322_w322_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x322x322` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x322x322x6 |
| `b1_c3_h324_w324_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x324x324` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x324x324x6 |
| `b1_c3_h326_w326_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x326x326` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x326x326x6 |
| `b1_c3_h328_w328_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x328x328` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x328x328x6 |
| `b1_c3_h330_w330_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x330x330` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x330x330x6 |
| `b1_c3_h332_w332_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x332x332` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x332x332x6 |
| `b1_c3_h334_w334_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x334x334` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x334x334x6 |
| `b1_c3_h336_w336_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x336x336` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x336x336x6 |
| `b1_c3_h338_w338_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x338x338` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x338x338x6 |
| `b1_c3_h340_w340_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x340x340` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x340x340x6 |
| `b1_c3_h342_w342_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x342x342` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x342x342x6 |
| `b1_c3_h344_w344_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x344x344` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x344x344x6 |
| `b1_c3_h346_w346_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x346x346` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x346x346x6 |
| `b1_c3_h348_w348_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x348x348` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x348x348x6 |
| `b1_c3_h350_w350_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x350x350` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x350x350x6 |
| `b1_c3_h352_w352_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x352x352` | `6x3x1x1` | 1 | PASS: max_diff=0.0035 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x352x352x6 |
| `b1_c3_h354_w354_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x354x354` | `6x3x1x1` | 1 | PASS: max_diff=0.0036 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x354x354x6 |
| `b1_c3_h356_w356_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x356x356` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x356x356x6 |
| `b1_c3_h358_w358_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x358x358` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x358x358x6 |
| `b1_c3_h360_w360_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x360x360` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x360x360x6 |
| `b1_c3_h362_w362_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x362x362` | `6x3x1x1` | 1 | PASS: max_diff=0.0037 | ERROR: ValueError: buffer is smaller than requested size sec=0.02 | PASS: PASS max_diff=1.0 out=1x362x362x6 |
| `b1_c3_h364_w364_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x364x364` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x364x364x6 |
| `b1_c3_h366_w366_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x366x366` | `6x3x1x1` | 1 | PASS: max_diff=0.0029 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x366x366x6 |
| `b1_c3_h368_w368_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x368x368` | `6x3x1x1` | 1 | PASS: max_diff=0.0035 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x368x368x6 |
| `b1_c3_h370_w370_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x370x370` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.56 | PASS: PASS max_diff=1.0 out=1x370x370x6 |
| `b1_c3_h372_w372_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x372x372` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.53 | PASS: PASS max_diff=1.0 out=1x372x372x6 |
| `b1_c3_h374_w374_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x374x374` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x374x374x6 |
| `b1_c3_h376_w376_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x376x376` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.60 | PASS: PASS max_diff=1.0 out=1x376x376x6 |
| `b1_c3_h378_w378_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x378x378` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x378x378x6 |
| `b1_c3_h380_w380_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x380x380` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x380x380x6 |
| `b1_c3_h382_w382_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x382x382` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x382x382x6 |
| `b1_c3_h384_w384_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x384x384` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.60 | PASS: PASS max_diff=1.0 out=1x384x384x6 |
| `b1_c3_h386_w386_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x386x386` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x386x386x6 |
| `b1_c3_h388_w388_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x388x388` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x388x388x6 |
| `b1_c3_h390_w390_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x390x390` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x390x390x6 |
| `b1_c3_h392_w392_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x392x392` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.60 | PASS: PASS max_diff=1.0 out=1x392x392x6 |
| `b1_c3_h394_w394_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x394x394` | `6x3x1x1` | 1 | PASS: max_diff=0.0039 | ERROR: ValueError: buffer is smaller than requested size sec=0.55 | PASS: PASS max_diff=1.0 out=1x394x394x6 |
| `b1_c3_h396_w396_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x396x396` | `6x3x1x1` | 1 | PASS: max_diff=0.0038 | ERROR: ValueError: buffer is smaller than requested size sec=0.54 | PASS: PASS max_diff=1.0 out=1x396x396x6 |
| `b1_c3_h398_w398_oc6_wic3_k1x1_g1_s1_pvalid` | `conv.py` | `1x3x398x398` | `6x3x1x1` | 1 | PASS: max_diff=0.0020 | ERROR: ValueError: buffer is smaller than requested size sec=0.58 | PASS: PASS max_diff=1.0 out=1x398x398x6 |
| `b1_c32_h112_w112_oc16_wic32_k1x1_g1_s1_pvalid` | `mesa.md` | `1x32x112x112` | `16x32x1x1` | 1 | PASS: max_diff=0.0146 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=56.1386 | PASS: PASS max_diff=1.0 out=1x112x112x16 |
| `b1_c16_h112_w112_oc96_wic16_k1x1_g1_s1_pvalid` | `mesa.md` | `1x16x112x112` | `96x16x1x1` | 1 | PASS: max_diff=0.0078 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=31.8750 | PASS: PASS max_diff=1.0 out=1x112x112x96 |
| `b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid` | `mesa.md` | `1x96x112x112` | `96x1x3x3` | 96 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=29.4768 | PASS: PASS max_diff=1.0 out=1x110x110x96 |
| `b1_c96_h56_w56_oc24_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x56x56` | `24x96x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x56x56x24 |
| `b1_c24_h56_w56_oc144_wic24_k1x1_g1_s1_pvalid` | `mesa.md` | `1x24x56x56` | `144x24x1x1` | 1 | PASS: max_diff=0.0078 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=48.9942 | FAIL: FAIL max_diff=255.0 out=1x56x56x144 |
| `b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid` | `mesa.md` | `1x144x56x56` | `144x1x3x3` | 144 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max | PASS: PASS max_diff=1.0 out=1x54x54x144 |
| `b1_c144_h56_w56_oc24_wic144_k1x1_g1_s1_pvalid` | `mesa.md` | `1x144x56x56` | `24x144x1x1` | 1 | PASS: max_diff=0.0418 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x56x56x24 |
| `b1_c144_h28_w28_oc32_wic144_k1x1_g1_s1_pvalid` | `mesa.md` | `1x144x28x28` | `32x144x1x1` | 1 | PASS: max_diff=0.0418 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x32 |
| `b1_c32_h28_w28_oc192_wic32_k1x1_g1_s1_pvalid` | `mesa.md` | `1x32x28x28` | `192x32x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=35.1260 | PASS: PASS max_diff=1.0 out=1x28x28x192 |
| `b1_c192_h28_w28_oc192_wic1_k3x3_g192_s1_pvalid` | `mesa.md` | `1x192x28x28` | `192x1x3x3` | 192 | PASS: max_diff=0.0063 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x26x26x192 |
| `b1_c192_h28_w28_oc32_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x28x28` | `32x192x1x1` | 1 | PASS: max_diff=0.0486 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x28x28x32 |
| `b1_c192_h14_w14_oc64_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x14x14` | `64x192x1x1` | 1 | PASS: max_diff=0.0292 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x64 |
| `b1_c64_h14_w14_oc384_wic64_k1x1_g1_s1_pvalid` | `mesa.md` | `1x64x14x14` | `384x64x1x1` | 1 | PASS: max_diff=0.0155 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=52.9608 | PASS: PASS max_diff=1.0 out=1x14x14x384 |
| `b1_c384_h14_w14_oc384_wic1_k3x3_g384_s1_pvalid` | `mesa.md` | `1x384x14x14` | `384x1x3x3` | 384 | PASS: max_diff=0.0039 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x12x12x384 |
| `b1_c384_h14_w14_oc64_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x14x14` | `64x384x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x64 |
| `b1_c384_h14_w14_oc96_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x14x14` | `96x384x1x1` | 1 | PASS: max_diff=0.0000 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x96 |
| `b1_c96_h14_w14_oc576_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x14x14` | `576x96x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x576 |
| `b1_c576_h14_w14_oc576_wic1_k3x3_g576_s1_pvalid` | `mesa.md` | `1x576x14x14` | `576x1x3x3` | 576 | PASS: max_diff=0.0071 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x12x12x576 |
| `b1_c576_h14_w14_oc96_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x14x14` | `96x576x1x1` | 1 | PASS: max_diff=0.0000 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x96 |
| `b1_c576_h7_w7_oc160_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x7x7` | `160x576x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x160 |
| `b1_c160_h7_w7_oc960_wic160_k1x1_g1_s1_pvalid` | `mesa.md` | `1x160x7x7` | `960x160x1x1` | 1 | PASS: max_diff=0.0280 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x960 |
| `b1_c960_h7_w7_oc960_wic1_k3x3_g960_s1_pvalid` | `mesa.md` | `1x960x7x7` | `960x1x3x3` | 960 | PASS: max_diff=0.0056 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x5x5x960 |
| `b1_c960_h7_w7_oc160_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x7x7` | `160x960x1x1` | 1 | PASS: max_diff=0.0561 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x160 |
| `b1_c960_h7_w7_oc320_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x7x7` | `320x960x1x1` | 1 | PASS: max_diff=0.0612 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x320 |
| `b1_c320_h7_w7_oc1280_wic320_k1x1_g1_s1_pvalid` | `mesa.md` | `1x320x7x7` | `1280x320x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x1280 |
| `b1_c1280_h1_w1_oc1001_wic1280_k1x1_g1_s1_pvalid` | `mesa.md` | `1x1280x1x1` | `1001x1280x1x1` | 1 | PASS: max_diff=0.0322 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x1x1x1001 |
| `b1_c3_h224_w224_oc64_wic3_k7x7_g1_s1_pvalid` | `mesa.md` | `1x3x224x224` | `64x3x7x7` | 1 | PASS: max_diff=0.0156 | FAIL: FAIL max_diff=205.0881 | PASS: PASS max_diff=1.0 out=1x218x218x64 |
| `b1_c64_h56_w56_oc64_wic64_k1x1_g1_s1_pvalid` | `mesa.md` | `1x64x56x56` | `64x64x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=75.1040 | PASS: PASS max_diff=1.0 out=1x56x56x64 |
| `b1_c64_h56_w56_oc192_wic64_k3x3_g1_s1_pvalid` | `mesa.md` | `1x64x56x56` | `192x64x3x3` | 1 | PASS: max_diff=0.0156 | FAIL: FAIL max_diff=688.0314 | PASS: PASS max_diff=1.0 out=1x54x54x192 |
| `b1_c192_h28_w28_oc64_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x28x28` | `64x192x1x1` | 1 | PASS: max_diff=0.0306 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x64 |
| `b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x28x28` | `96x192x1x1` | 1 | FAIL: max_diff=116.0390 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x28x28x96 |
| `b1_c96_h28_w28_oc128_wic96_k3x3_g1_s1_pvalid` | `mesa.md` | `1x96x28x28` | `128x96x3x3` | 1 | PASS: max_diff=0.0078 | FAIL: FAIL max_diff=172.5735 | PASS: PASS max_diff=0.0 out=1x26x26x128 |
| `b1_c192_h28_w28_oc16_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x28x28` | `16x192x1x1` | 1 | PASS: max_diff=0.0486 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x16 |
| `b1_c16_h28_w28_oc32_wic16_k3x3_g1_s1_pvalid` | `mesa.md` | `1x16x28x28` | `32x16x3x3` | 1 | PASS: max_diff=0.0039 | FAIL: FAIL max_diff=65.2610 | PASS: PASS max_diff=0.0 out=1x26x26x32 |
| `b1_c256_h28_w28_oc128_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x28x28` | `128x256x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x128 |
| `b1_c128_h28_w28_oc192_wic128_k3x3_g1_s1_pvalid` | `mesa.md` | `1x128x28x28` | `192x128x3x3` | 1 | PASS: max_diff=0.0156 | FAIL: FAIL max_diff=210.8453 | PASS: PASS max_diff=0.0 out=1x26x26x192 |
| `b1_c256_h28_w28_oc32_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x28x28` | `32x256x1x1` | 1 | PASS: max_diff=0.0735 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x32 |
| `b1_c32_h28_w28_oc96_wic32_k3x3_g1_s1_pvalid` | `mesa.md` | `1x32x28x28` | `96x32x3x3` | 1 | PASS: max_diff=0.0039 | FAIL: FAIL max_diff=109.0997 | PASS: PASS max_diff=0.0 out=1x26x26x96 |
| `b1_c256_h28_w28_oc64_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x28x28` | `64x256x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x28x28x64 |
| `b1_c480_h14_w14_oc192_wic480_k1x1_g1_s1_pvalid` | `mesa.md` | `1x480x14x14` | `192x480x1x1` | 1 | PASS: max_diff=0.0324 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x192 |
| `b1_c480_h14_w14_oc96_wic480_k1x1_g1_s1_pvalid` | `mesa.md` | `1x480x14x14` | `96x480x1x1` | 1 | PASS: max_diff=0.0000 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x96 |
| `b1_c96_h14_w14_oc208_wic96_k3x3_g1_s1_pvalid` | `mesa.md` | `1x96x14x14` | `208x96x3x3` | 1 | PASS: max_diff=0.0039 | FAIL: FAIL max_diff=139.7270 | PASS: PASS max_diff=0.0 out=1x12x12x208 |
| `b1_c480_h14_w14_oc16_wic480_k1x1_g1_s1_pvalid` | `mesa.md` | `1x480x14x14` | `16x480x1x1` | 1 | PASS: max_diff=0.0937 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x16 |
| `b1_c16_h14_w14_oc48_wic16_k3x3_g1_s1_pvalid` | `mesa.md` | `1x16x14x14` | `48x16x3x3` | 1 | PASS: max_diff=0.0039 | FAIL: FAIL max_diff=55.9387 | PASS: PASS max_diff=0.0 out=1x12x12x48 |
| `b1_c480_h14_w14_oc64_wic480_k1x1_g1_s1_pvalid` | `mesa.md` | `1x480x14x14` | `64x480x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x64 |
| `b1_c512_h14_w14_oc160_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `160x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x160 |
| `b1_c512_h14_w14_oc112_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `112x512x1x1` | 1 | PASS: max_diff=0.0000 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x112 |
| `b1_c112_h14_w14_oc224_wic112_k3x3_g1_s1_pvalid` | `mesa.md` | `1x112x14x14` | `224x112x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=168.3972 | FAIL: FAIL max_diff=255.0 out=1x12x12x224 |
| `b1_c512_h14_w14_oc24_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `24x512x1x1` | 1 | PASS: max_diff=0.1038 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x24 |
| `b1_c24_h14_w14_oc64_wic24_k3x3_g1_s1_pvalid` | `mesa.md` | `1x24x14x14` | `64x24x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=74.4538 | FAIL: FAIL max_diff=255.0 out=1x12x12x64 |
| `b1_c512_h14_w14_oc64_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `64x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x64 |
| `b1_c512_h14_w14_oc128_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `128x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x128 |
| `b1_c128_h14_w14_oc256_wic128_k3x3_g1_s1_pvalid` | `mesa.md` | `1x128x14x14` | `256x128x3x3` | 1 | PASS: max_diff=0.0156 | FAIL: FAIL max_diff=149.5481 | PASS: PASS max_diff=0.0 out=1x12x12x256 |
| `b1_c512_h14_w14_oc144_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `144x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x144 |
| `b1_c144_h14_w14_oc288_wic144_k3x3_g1_s1_pvalid` | `mesa.md` | `1x144x14x14` | `288x144x3x3` | 1 | PASS: max_diff=0.0156 | FAIL: FAIL max_diff=170.8285 | PASS: PASS max_diff=0.0 out=1x12x12x288 |
| `b1_c512_h14_w14_oc32_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x14x14` | `32x512x1x1` | 1 | PASS: max_diff=0.1038 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x32 |
| `b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid` | `mesa.md` | `1x32x14x14` | `64x32x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=78.7266 | PASS: PASS max_diff=0.0 out=1x12x12x64 |
| `b1_c528_h14_w14_oc256_wic528_k1x1_g1_s1_pvalid` | `mesa.md` | `1x528x14x14` | `256x528x1x1` | 1 | FAIL: max_diff=203.4367 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x256 |
| `b1_c528_h14_w14_oc160_wic528_k1x1_g1_s1_pvalid` | `mesa.md` | `1x528x14x14` | `160x528x1x1` | 1 | FAIL: max_diff=203.4367 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x160 |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | `mesa.md` | `1x160x14x14` | `320x160x3x3` | 1 | SKIP: mesa_semantics | TIMEOUT: timeout_20s | PASS: PASS max_diff=0.0 out=1x12x12x320 |
| `b1_c528_h14_w14_oc32_wic528_k1x1_g1_s1_pvalid` | `mesa.md` | `1x528x14x14` | `32x528x1x1` | 1 | FAIL: max_diff=166.0385 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x32 |
| `b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid` | `mesa.md` | `1x32x14x14` | `128x32x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=78.7266 | PASS: PASS max_diff=0.0 out=1x12x12x128 |
| `b1_c528_h14_w14_oc128_wic528_k1x1_g1_s1_pvalid` | `mesa.md` | `1x528x14x14` | `128x528x1x1` | 1 | FAIL: max_diff=203.4367 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x14x14x128 |
| `b1_c832_h7_w7_oc256_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `256x832x1x1` | 1 | PASS: max_diff=0.0403 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x256 |
| `b1_c832_h7_w7_oc160_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `160x832x1x1` | 1 | PASS: max_diff=0.0403 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x160 |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid` | `mesa.md` | `1x160x7x7` | `320x160x3x3` | 1 | SKIP: mesa_semantics | TIMEOUT: timeout_20s | PASS: PASS max_diff=0.0 out=1x5x5x320 |
| `b1_c832_h7_w7_oc32_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `32x832x1x1` | 1 | PASS: max_diff=0.0311 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x32 |
| `b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid` | `mesa.md` | `1x32x7x7` | `128x32x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=72.8922 | PASS: PASS max_diff=0.0 out=1x5x5x128 |
| `b1_c832_h7_w7_oc128_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `128x832x1x1` | 1 | PASS: max_diff=0.0382 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x128 |
| `b1_c832_h7_w7_oc384_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `384x832x1x1` | 1 | PASS: max_diff=0.0505 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x384 |
| `b1_c832_h7_w7_oc192_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `192x832x1x1` | 1 | PASS: max_diff=0.0403 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x192 |
| `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid` | `mesa.md` | `1x192x7x7` | `384x192x3x3` | 1 | SKIP: mesa_semantics | TIMEOUT: timeout_20s | PASS: PASS max_diff=0.0 out=1x5x5x384 |
| `b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid` | `mesa.md` | `1x832x7x7` | `48x832x1x1` | 1 | FAIL: max_diff=185.9685 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x7x7x48 |
| `b1_c48_h7_w7_oc128_wic48_k3x3_g1_s1_pvalid` | `mesa.md` | `1x48x7x7` | `128x48x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=112.9228 | FAIL: FAIL max_diff=255.0 out=1x5x5x128 |
| `b0_c0_h0_w0_oc32_wic3_k3x3_g1_s1_pvalid` | `mesa.md` | `0x0x0x0` | `32x3x3x3` | 1 | SKIP: mesa_semantics | UNSUPPORTED: UNSUPPORTED batch | UNSUPPORTED: batch |
| `b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid` | `mesa.md` | `1x32x150x150` | `32x1x3x3` | 32 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=25.9643 | PASS: PASS max_diff=1.0 out=1x148x148x32 |
| `b1_c32_h150_w150_oc16_wic32_k1x1_g1_s1_pvalid` | `mesa.md` | `1x32x150x150` | `16x32x1x1` | 1 | FAIL: max_diff=91.2539 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=54.7856 | PASS: PASS max_diff=1.0 out=1x150x150x16 |
| `b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid` | `mesa.md` | `1x16x150x150` | `96x16x1x1` | 1 | ERROR: max_diff= | ERROR: SUBMIT ret=0; Traceback (most recent call last):;   File "/home/orangepi/rk3588/experimental/mainline6_18/conv_mesa_shape_once.py", line 102, in <module>;     raise SystemExit(main());   File "/home/orangepi/rk3588/experimental/mainline6_18/conv_mesa_shape_onc | PASS: PASS max_diff=1.0 out=1x150x150x96 |
| `b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid` | `mesa.md` | `1x96x150x150` | `96x1x3x3` | 96 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=30.4802 | PASS: PASS max_diff=1.0 out=1x148x148x96 |
| `b1_c96_h75_w75_oc24_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x75x75` | `24x96x1x1` | 1 | FAIL: max_diff=93.6123 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x75x75x24 |
| `b1_c24_h75_w75_oc144_wic24_k1x1_g1_s1_pvalid` | `mesa.md` | `1x24x75x75` | `144x24x1x1` | 1 | PASS: max_diff=0.0078 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=39.9441 | FAIL: FAIL max_diff=255.0 out=1x75x75x144 |
| `b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid` | `mesa.md` | `1x144x75x75` | `144x1x3x3` | 144 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max | PASS: PASS max_diff=1.0 out=1x73x73x144 |
| `b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid` | `mesa.md` | `1x144x75x75` | `24x144x1x1` | 1 | FAIL: max_diff=95.9232 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x75x75x24 |
| `b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid` | `mesa.md` | `1x144x38x38` | `32x144x1x1` | 1 | FAIL: max_diff=108.1646 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x38x38x32 |
| `b1_c32_h38_w38_oc192_wic32_k1x1_g1_s1_pvalid` | `mesa.md` | `1x32x38x38` | `192x32x1x1` | 1 | PASS: max_diff=0.0120 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=59.7625 | PASS: PASS max_diff=1.0 out=1x38x38x192 |
| `b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid` | `mesa.md` | `1x192x38x38` | `192x1x3x3` | 192 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x36x36x192 |
| `b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x38x38` | `32x192x1x1` | 1 | FAIL: max_diff=101.8545 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x38x38x32 |
| `b1_c192_h19_w19_oc64_wic192_k1x1_g1_s1_pvalid` | `mesa.md` | `1x192x19x19` | `64x192x1x1` | 1 | PASS: max_diff=0.0307 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x64 |
| `b1_c64_h19_w19_oc384_wic64_k1x1_g1_s1_pvalid` | `mesa.md` | `1x64x19x19` | `384x64x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=52.4223 | PASS: PASS max_diff=1.0 out=1x19x19x384 |
| `b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid` | `mesa.md` | `1x384x19x19` | `384x1x3x3` | 384 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x17x17x384 |
| `b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x19x19` | `64x384x1x1` | 1 | FAIL: max_diff=131.4672 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x64 |
| `b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x19x19` | `96x384x1x1` | 1 | FAIL: max_diff=151.9435 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x96 |
| `b1_c96_h19_w19_oc576_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x19x19` | `576x96x1x1` | 1 | PASS: max_diff=0.0193 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x19x19x576 |
| `b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid` | `mesa.md` | `1x576x19x19` | `576x1x3x3` | 576 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x17x17x576 |
| `b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x19x19` | `96x576x1x1` | 1 | FAIL: max_diff=215.2579 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x96 |
| `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x19x19` | `12x576x1x1` | 1 | FAIL: max_diff=268.2881 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x12 |
| `b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x19x19` | `273x576x1x1` | 1 | FAIL: max_diff=207.2943 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x19x19x273 |
| `b1_c576_h10_w10_oc160_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x10x10` | `160x576x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x10x10x160 |
| `b1_c160_h10_w10_oc960_wic160_k1x1_g1_s1_pvalid` | `mesa.md` | `1x160x10x10` | `960x160x1x1` | 1 | PASS: max_diff=0.0215 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x10x10x960 |
| `b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid` | `mesa.md` | `1x960x10x10` | `960x1x3x3` | 960 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x8x8x960 |
| `b1_c960_h10_w10_oc160_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x10x10` | `160x960x1x1` | 1 | PASS: max_diff=0.0617 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x160 |
| `b1_c960_h10_w10_oc320_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x10x10` | `320x960x1x1` | 1 | PASS: max_diff=0.0617 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x320 |
| `b1_c320_h10_w10_oc1280_wic320_k1x1_g1_s1_pvalid` | `mesa.md` | `1x320x10x10` | `1280x320x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x10x10x1280 |
| `b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid` | `mesa.md` | `1x1280x10x10` | `24x1280x1x1` | 1 | FAIL: max_diff=268.3582 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x24 |
| `b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid` | `mesa.md` | `1x1280x10x10` | `546x1280x1x1` | 1 | FAIL: max_diff=283.1043 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x546 |
| `b1_c1280_h10_w10_oc256_wic1280_k1x1_g1_s1_pvalid` | `mesa.md` | `1x1280x10x10` | `256x1280x1x1` | 1 | PASS: max_diff=0.0618 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x256 |
| `b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid` | `mesa.md` | `1x256x10x10` | `512x256x3x3` | 1 | SKIP: mesa_semantics | TIMEOUT: timeout_20s | PASS: PASS max_diff=1.0 out=1x8x8x512 |
| `b1_c512_h5_w5_oc24_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x5x5` | `24x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x5x5x24 |
| `b1_c512_h5_w5_oc546_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x5x5` | `546x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x5x5x546 |
| `b1_c512_h5_w5_oc128_wic512_k1x1_g1_s1_pvalid` | `mesa.md` | `1x512x5x5` | `128x512x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x5x5x128 |
| `b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid` | `mesa.md` | `1x128x5x5` | `256x128x3x3` | 1 | FAIL: max_diff=251.8077 | FAIL: FAIL max_diff=153.8202 | PASS: PASS max_diff=0.0 out=1x3x3x256 |
| `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x3x3` | `24x256x1x1` | 1 | FAIL: max_diff=88.1959 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x3x3x24 |
| `b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x3x3` | `546x256x1x1` | 1 | FAIL: max_diff=120.4964 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x3x3x546 |
| `b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x3x3` | `128x256x1x1` | 1 | FAIL: max_diff=120.4964 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x3x3x128 |
| `b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid` | `mesa.md` | `1x128x3x3` | `256x128x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=0.5869 | PASS: PASS max_diff=0.0 out=1x1x1x256 |
| `b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x2x2` | `24x256x1x1` | 1 | FAIL: max_diff=75.0104 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x2x2x24 |
| `b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x2x2` | `546x256x1x1` | 1 | FAIL: max_diff=94.8168 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x2x2x546 |
| `b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x2x2` | `64x256x1x1` | 1 | FAIL: max_diff=75.0104 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x2x2x64 |
| `b1_c64_h2_w2_oc128_wic64_k3x3_g1_s1_pvalid` | `mesa.md` | `1x64x2x2` | `128x64x3x3` | 1 | SKIP: mesa_semantics | UNSUPPORTED: UNSUPPORTED invalid_output | ERROR:  |
| `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x1x1` | `24x128x1x1` | 1 | FAIL: max_diff=55.4996 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x1x1x24 |
| `b1_c128_h1_w1_oc546_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x1x1` | `546x128x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x1x1x546 |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | `mesa.md` | `1x3x320x320` | `32x3x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=41.1484 | PASS: PASS max_diff=1.0 out=1x318x318x32 |
| `b1_c32_h160_w160_oc8_wic32_k1x1_g1_s1_pvalid` | `mesa.md` | `1x32x160x160` | `8x32x1x1` | 1 | FAIL: max_diff=80.1930 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=49.4966 | PASS: PASS max_diff=0.0 out=1x160x160x8 |
| `b1_c8_h160_w160_oc16_wic8_k3x3_g1_s1_pvalid` | `mesa.md` | `1x8x160x160` | `16x8x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=94.1173 | PASS: PASS max_diff=1.0 out=1x158x158x16 |
| `b1_c16_h160_w160_oc16_wic16_k1x1_g1_s1_pvalid` | `mesa.md` | `1x16x160x160` | `16x16x1x1` | 1 | PASS: max_diff=0.0078 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=23.5342 | PASS: PASS max_diff=1.0 out=1x160x160x16 |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | `mesa.md` | `1x16x160x160` | `128x16x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=224.3309 | PASS: PASS max_diff=1.0 out=1x158x158x128 |
| `b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x80x80` | `16x128x1x1` | 1 | FAIL: max_diff=88.6326 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x80x80x16 |
| `b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid` | `mesa.md` | `1x16x80x80` | `64x16x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=224.1263 | PASS: PASS max_diff=1.0 out=1x78x78x64 |
| `b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid` | `mesa.md` | `1x64x80x80` | `16x64x1x1` | 1 | FAIL: max_diff=77.7067 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=87.6969 | PASS: PASS max_diff=1.0 out=1x80x80x16 |
| `b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` | `mesa.md` | `1x16x80x80` | `128x16x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=224.1263 | PASS: PASS max_diff=1.0 out=1x78x78x128 |
| `b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid` | `mesa.md` | `1x16x80x80` | `128x16x5x5` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=508.5211 | PASS: PASS max_diff=1.0 out=1x76x76x128 |
| `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x40x40` | `40x128x1x1` | 1 | FAIL: max_diff=97.6727 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x40x40x40 |
| `b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid` | `mesa.md` | `1x40x40x40` | `160x40x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=449.0205 | PASS: PASS max_diff=1.0 out=1x38x38x160 |
| `b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid` | `mesa.md` | `1x160x40x40` | `40x160x1x1` | 1 | FAIL: max_diff=100.3703 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x40x40x40 |
| `b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` | `mesa.md` | `1x40x40x40` | `320x40x1x1` | 1 | FAIL: max_diff=54.1734 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=72.5378 | PASS: PASS max_diff=1.0 out=1x40x40x320 |
| `b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid` | `mesa.md` | `1x320x40x40` | `320x1x3x3` | 320 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x38x38x320 |
| `b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid` | `mesa.md` | `1x320x20x20` | `72x320x1x1` | 1 | FAIL: max_diff=148.1625 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x72 |
| `b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid` | `mesa.md` | `1x72x20x20` | `576x72x1x1` | 1 | FAIL: max_diff=70.0652 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max | PASS: PASS max_diff=1.0 out=1x20x20x576 |
| `b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid` | `mesa.md` | `1x576x20x20` | `576x1x3x3` | 576 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x18x18x576 |
| `b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x20x20` | `72x576x1x1` | 1 | FAIL: max_diff=160.2361 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x72 |
| `b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid` | `mesa.md` | `1x72x20x20` | `288x72x3x3` | 1 | SKIP: mesa_semantics | FAIL: FAIL max_diff=160.4444 | PASS: PASS max_diff=0.0 out=1x18x18x288 |
| `b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid` | `mesa.md` | `1x288x20x20` | `72x288x1x1` | 1 | FAIL: max_diff=128.0329 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x72 |
| `b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid` | `mesa.md` | `1x576x20x20` | `576x1x5x5` | 576 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x16x16x576 |
| `b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid` | `mesa.md` | `1x576x20x20` | `96x576x1x1` | 1 | FAIL: max_diff=179.1007 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x96 |
| `b1_c96_h20_w20_oc768_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x20x20` | `768x96x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x20x20x768 |
| `b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid` | `mesa.md` | `1x768x20x20` | `768x1x5x5` | 768 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x16x16x768 |
| `b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid` | `mesa.md` | `1x768x20x20` | `96x768x1x1` | 1 | FAIL: max_diff=181.1810 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x96 |
| `b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid` | `mesa.md` | `1x768x20x20` | `768x1x3x3` | 768 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x18x18x768 |
| `b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid` | `mesa.md` | `1x768x10x10` | `120x768x1x1` | 1 | FAIL: max_diff=210.0372 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x120 |
| `b1_c120_h10_w10_oc960_wic120_k1x1_g1_s1_pvalid` | `mesa.md` | `1x120x10x10` | `960x120x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x10x10x960 |
| `b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x10x10` | `120x960x1x1` | 1 | FAIL: max_diff=229.2946 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x120 |
| `b1_c120_h10_w10_oc480_wic120_k1x1_g1_s1_pvalid` | `mesa.md` | `1x120x10x10` | `480x120x1x1` | 1 | PASS: max_diff=0.0156 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x10x10x480 |
| `b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid` | `mesa.md` | `1x480x10x10` | `480x1x5x5` | 480 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x6x6x480 |
| `b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid` | `mesa.md` | `1x480x10x10` | `120x480x1x1` | 1 | FAIL: max_diff=162.1157 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x120 |
| `b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid` | `mesa.md` | `1x960x10x10` | `960x1x5x5` | 960 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x6x6x960 |
| `b1_c960_h10_w10_oc384_wic960_k1x1_g1_s1_pvalid` | `mesa.md` | `1x960x10x10` | `384x960x1x1` | 1 | PASS: max_diff=0.0617 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x384 |
| `b1_c384_h10_w10_oc256_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x10x10` | `256x384x1x1` | 1 | PASS: max_diff=0.0312 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x256 |
| `b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid` | `mesa.md` | `1x256x10x10` | `256x1x3x3` | 256 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x8x8x256 |
| `b1_c256_h5_w5_oc512_wic256_k1x1_g1_s1_pvalid` | `mesa.md` | `1x256x5x5` | `512x256x1x1` | 1 | PASS: max_diff=0.0298 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x5x5x512 |
| `b1_c128_h5_w5_oc128_wic1_k3x3_g128_s1_pvalid` | `mesa.md` | `1x128x5x5` | `128x1x3x3` | 128 | PASS: max_diff=0.0039 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=18.7847 | PASS: PASS max_diff=0.0 out=1x3x3x128 |
| `b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x3x3` | `256x128x1x1` | 1 | FAIL: max_diff=67.4651 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x3x3x256 |
| `b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid` | `mesa.md` | `1x128x3x3` | `128x1x3x3` | 128 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=11.5886 | PASS: PASS max_diff=0.0 out=1x1x1x128 |
| `b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid` | `mesa.md` | `1x128x2x2` | `256x128x1x1` | 1 | FAIL: max_diff=69.6704 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x2x2x256 |
| `b1_c64_h2_w2_oc64_wic1_k3x3_g64_s1_pvalid` | `mesa.md` | `1x64x2x2` | `64x1x3x3` | 64 | SKIP: mesa_semantics | UNSUPPORTED: UNSUPPORTED invalid_output | ERROR:  |
| `b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid` | `mesa.md` | `1x64x1x1` | `128x64x1x1` | 1 | FAIL: max_diff=39.0726 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=20.7031 | PASS: PASS max_diff=0.0 out=1x1x1x128 |
| `b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid` | `mesa.md` | `1x96x20x20` | `96x1x3x3` | 96 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; FAIL max_diff=20.6459 | PASS: PASS max_diff=1.0 out=1x18x18x96 |
| `b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid` | `mesa.md` | `1x384x10x10` | `384x1x3x3` | 384 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x8x8x384 |
| `b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid` | `mesa.md` | `1x512x5x5` | `512x1x3x3` | 512 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x3x3x512 |
| `b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid` | `mesa.md` | `1x256x3x3` | `256x1x3x3` | 256 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x1x1x256 |
| `b1_c256_h2_w2_oc256_wic1_k3x3_g256_s1_pvalid` | `mesa.md` | `1x256x2x2` | `256x1x3x3` | 256 | SKIP: mesa_semantics | UNSUPPORTED: UNSUPPORTED invalid_output | ERROR:  |
| `b1_c128_h1_w1_oc128_wic1_k3x3_g128_s1_pvalid` | `mesa.md` | `1x128x1x1` | `128x1x3x3` | 128 | SKIP: mesa_semantics | UNSUPPORTED: UNSUPPORTED invalid_output | ERROR: Traceback (most recent call last):;   File "<string>", line 5, in <module>;   File "/home/orangepi/mesa/.venv/lib/python3.10/site-packages/tflite_runtime/interp |
| `b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x20x20` | `12x96x1x1` | 1 | SKIP: mesa_semantics | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x20x20x12 |
| `b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid` | `mesa.md` | `1x96x20x20` | `273x96x1x1` | 1 | FAIL: max_diff=100.7007 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=1.0 out=1x20x20x273 |
| `b1_c384_h10_w10_oc24_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x10x10` | `24x384x1x1` | 1 | PASS: max_diff=0.0286 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x24 |
| `b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid` | `mesa.md` | `1x384x10x10` | `546x384x1x1` | 1 | FAIL: max_diff=165.6544 | ERROR: SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT ret=0; SUBMIT r | PASS: PASS max_diff=0.0 out=1x10x10x546 |
