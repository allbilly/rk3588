"""Copy FP16 tensor representations between canonical and native layouts.

These routines move bytes and insert padding. They do not perform model tensor
arithmetic on the CPU. Canonical tensors follow the queried logical format.
"""


def pack(raw, logical, native):
    if len(raw) != logical["elements"] * 2:
        raise ValueError("Canonical FP16 input has the wrong element count")
    size = native["size_with_stride"] or native["size"]
    if native["format"] == 2:
        n, c1, h, w, c2 = native["dims"]
        channels = logical["elements"] // (n * h * w)
        if channels * n * h * w != logical["elements"]:
            raise ValueError("Native shape cannot represent the logical tensor")
        stride = native["w_stride"] or w
        if h == w == stride == 1 and channels == c1 * c2:
            return raw + bytes(size - len(raw))
        result = bytearray(size)
        for batch in range(n):
            for channel in range(channels):
                group, lane = divmod(channel, c2)
                for y in range(h):
                    source = (((batch * h + y) * w * channels + channel)
                              if logical["format"] == 1 else ((batch * channels + channel) * h + y) * w)
                    step = channels * 2 if logical["format"] == 1 else 2
                    destination = (((batch * c1 + group) * h + y) * stride * c2 + lane) * 2
                    for byte in (0, 1):
                        result[destination + byte:destination + w * c2 * 2:c2 * 2] = raw[source * 2 + byte:source * 2 + w * step:step]
        return bytes(result)
    if native["format"] == 1 and logical["format"] in (0, 1) and len(logical["dims"]) == 4:
        if logical["format"] == 0:
            n, channels, h, w = logical["dims"]
        else:
            n, h, w, channels = logical["dims"]
        stride = native["w_stride"] or w
        if stride < w or n * h * stride * channels * 2 > size:
            raise ValueError("Native NHWC row stride does not fit its allocation")
        result = bytearray(size)
        if logical["format"] == 1:
            row_bytes = w * channels * 2
            for row in range(n * h):
                destination = row * stride * channels * 2
                result[destination:destination + row_bytes] = raw[row * row_bytes:(row + 1) * row_bytes]
            return bytes(result)
        for batch in range(n):
            for channel in range(channels):
                for y in range(h):
                    for x in range(w):
                        source = ((batch * channels + channel) * h + y) * w + x
                        destination = ((batch * h + y) * stride + x) * channels + channel
                        result[destination * 2:destination * 2 + 2] = raw[source * 2:source * 2 + 2]
        return bytes(result)
    return raw + bytes(size - len(raw))


def unpack(raw, logical, native):
    if native["format"] == 2:
        n, c1, h, w, c2 = native["dims"]
        channels = logical["elements"] // (n * h * w)
        stride = native["w_stride"] or w
        if h == w == stride == 1 and channels == c1 * c2:
            return raw[:logical["elements"] * 2]
        result = bytearray(logical["elements"] * 2)
        for batch in range(n):
            for channel in range(channels):
                group, lane = divmod(channel, c2)
                for y in range(h):
                    source = (((batch * c1 + group) * h + y) * stride * c2 + lane) * 2
                    destination = (((batch * h + y) * w * channels + channel)
                                   if logical["format"] == 1 else ((batch * channels + channel) * h + y) * w)
                    step = channels * 2 if logical["format"] == 1 else 2
                    for byte in (0, 1):
                        result[destination * 2 + byte:destination * 2 + w * step:step] = raw[source + byte:source + w * c2 * 2:c2 * 2]
        return bytes(result)
    if native["format"] == 1 and logical["format"] in (0, 1) and len(logical["dims"]) == 4:
        if logical["format"] == 0:
            n, channels, h, w = logical["dims"]
        else:
            n, h, w, channels = logical["dims"]
        stride = native["w_stride"] or w
        if stride < w or n * h * stride * channels * 2 > len(raw):
            raise ValueError("Native NHWC row stride does not fit its allocation")
        result = bytearray(logical["elements"] * 2)
        if logical["format"] == 1:
            row_bytes = w * channels * 2
            for row in range(n * h):
                source = row * stride * channels * 2
                result[row * row_bytes:(row + 1) * row_bytes] = raw[source:source + row_bytes]
            return bytes(result)
        for batch in range(n):
            for channel in range(channels):
                for y in range(h):
                    for x in range(w):
                        source = ((batch * h + y) * stride + x) * channels + channel
                        destination = ((batch * channels + channel) * h + y) * w + x
                        result[destination * 2:destination * 2 + 2] = raw[source * 2:source * 2 + 2]
        return bytes(result)
    return raw[:logical["elements"] * 2]
