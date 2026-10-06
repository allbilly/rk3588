"""Explicit byte-only transpose at a reference model's graph boundary."""
from itertools import product
from tensor_io import pack, unpack


def transpose(raw, shape, permutation):
    if sorted(permutation) != list(range(len(shape))):
        raise ValueError("Invalid transpose permutation")
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    output_shape = [shape[index] for index in permutation]
    result = bytearray(len(raw))
    width = output_shape[-1]
    step = strides[permutation[-1]] * 2
    row = 0
    for coordinates in product(*(range(size) for size in output_shape[:-1])):
        source = sum(value * strides[axis] for value, axis in zip(coordinates, permutation[:-1])) * 2
        for byte in (0, 1):
            result[row + byte:row + width * 2:2] = raw[source + byte:source + width * step:step]
        row += width * 2
    return bytes(result)


def apply(operation, buffers):
    if operation["kind"] != "transpose":
        raise ValueError("Unsupported layout operation")
    source = operation["source"]
    target = operation["destination"]
    buffer = buffers[source["memory"]["buffer"]]
    buffer.sync(2)
    offset = source["memory"]["offset"]
    raw = buffer.mapping[offset:offset + source["native"]["size_with_stride"]]
    canonical = unpack(raw, source["logical"], source["native"])
    moved = transpose(canonical, source["logical"]["dims"], operation["permutation"])
    packed = pack(moved, target["logical"], target["native"])
    buffer = buffers[target["memory"]["buffer"]]
    buffer.write(packed, target["memory"]["offset"])
    buffer.sync(1)
