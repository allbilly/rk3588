import ctypes
import os


ROOT = os.path.dirname(__file__)
LIB_CANDIDATES = [
    "/lib/librknnrt.so",
    "/usr/lib/librknnrt.so",
    os.path.join(ROOT, "..", "rknn-llm", "examples", "multimodal_model_demo", "deploy", "3rdparty",
                 "librknnrt", "Linux", "librknn_api", "aarch64", "librknnrt.so"),
    os.path.join(ROOT, "..", "rknn-toolkit2", "rknpu2", "runtime", "Linux", "librknn_api", "aarch64", "librknnrt.so"),
]


def _load_library():
    for path in LIB_CANDIDATES:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError("could not find librknnrt.so for matmul API")


lib = _load_library()

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_UINT16 = 5
RKNN_TENSOR_INT32 = 6
RKNN_TENSOR_UINT32 = 7
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_BOOL = 9
RKNN_TENSOR_INT4 = 10
RKNN_TENSOR_BFLOAT16 = 11

RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1
RKNN_INT8_MM_INT8_TO_INT32 = 2
RKNN_INT8_MM_INT8_TO_INT8 = 3
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4
RKNN_FLOAT16_MM_INT8_TO_FLOAT32 = 5
RKNN_FLOAT16_MM_INT8_TO_FLOAT16 = 6
RKNN_FLOAT16_MM_INT4_TO_FLOAT32 = 7
RKNN_FLOAT16_MM_INT4_TO_FLOAT16 = 8
RKNN_INT8_MM_INT8_TO_FLOAT32 = 9
RKNN_INT4_MM_INT4_TO_INT16 = 10
RKNN_INT8_MM_INT4_TO_INT32 = 11

RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0 = 1

RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2


class rknn_tensor_mem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class rknn_matmul_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("size", ctypes.c_uint32),
        ("type", ctypes.c_int32),
    ]


class rknn_matmul_io_attr(ctypes.Structure):
    _fields_ = [
        ("A", rknn_matmul_tensor_attr),
        ("B", rknn_matmul_tensor_attr),
        ("C", rknn_matmul_tensor_attr),
    ]


class rknn_matmul_info(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_int32),
        ("K", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("B_layout", ctypes.c_int16),
        ("B_quant_type", ctypes.c_int16),
        ("AC_layout", ctypes.c_int16),
        ("AC_quant_type", ctypes.c_int16),
        ("iommu_domain_id", ctypes.c_int32),
        ("group_size", ctypes.c_int16),
        ("reserved", ctypes.c_int8 * 34),
    ]


class rknn_quant_params(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("scale", ctypes.POINTER(ctypes.c_float)),
        ("scale_len", ctypes.c_int32),
        ("zp", ctypes.POINTER(ctypes.c_int32)),
        ("zp_len", ctypes.c_int32),
    ]


class rknn_matmul_shape(ctypes.Structure):
    _fields_ = [("M", ctypes.c_int32), ("K", ctypes.c_int32), ("N", ctypes.c_int32)]


rknn_matmul_ctx = ctypes.c_uint64


lib.rknn_matmul_create.argtypes = [ctypes.POINTER(rknn_matmul_ctx), ctypes.POINTER(rknn_matmul_info),
                                   ctypes.POINTER(rknn_matmul_io_attr)]
lib.rknn_matmul_create.restype = ctypes.c_int32
lib.rknn_matmul_create_dynamic_shape.argtypes = [ctypes.POINTER(rknn_matmul_ctx), ctypes.POINTER(rknn_matmul_info),
                                                 ctypes.c_int32, ctypes.POINTER(rknn_matmul_shape),
                                                 ctypes.POINTER(rknn_matmul_io_attr)]
lib.rknn_matmul_create_dynamic_shape.restype = ctypes.c_int32
lib.rknn_matmul_set_io_mem.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_tensor_mem),
                                       ctypes.POINTER(rknn_matmul_tensor_attr)]
lib.rknn_matmul_set_io_mem.restype = ctypes.c_int32
lib.rknn_matmul_set_core_mask.argtypes = [rknn_matmul_ctx, ctypes.c_int32]
lib.rknn_matmul_set_core_mask.restype = ctypes.c_int32
lib.rknn_matmul_set_quant_params.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_quant_params)]
lib.rknn_matmul_set_quant_params.restype = ctypes.c_int32
lib.rknn_matmul_run.argtypes = [rknn_matmul_ctx]
lib.rknn_matmul_run.restype = ctypes.c_int32
lib.rknn_matmul_destroy.argtypes = [rknn_matmul_ctx]
lib.rknn_matmul_destroy.restype = ctypes.c_int32
lib.rknn_create_mem.argtypes = [rknn_matmul_ctx, ctypes.c_uint32]
lib.rknn_create_mem.restype = ctypes.POINTER(rknn_tensor_mem)
lib.rknn_create_mem2.argtypes = [rknn_matmul_ctx, ctypes.c_uint64, ctypes.c_uint64]
lib.rknn_create_mem2.restype = ctypes.POINTER(rknn_tensor_mem)
lib.rknn_destroy_mem.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_tensor_mem)]
lib.rknn_destroy_mem.restype = ctypes.c_int32
lib.rknn_mem_sync.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_tensor_mem), ctypes.c_int32]
lib.rknn_mem_sync.restype = ctypes.c_int32


def make_info(m, k, n, matmul_type, b_layout=0, ac_layout=0, iommu_domain_id=0, group_size=0):
    info = rknn_matmul_info()
    info.M = m
    info.K = k
    info.N = n
    info.type = matmul_type
    info.B_layout = b_layout
    info.B_quant_type = 0
    info.AC_layout = ac_layout
    info.AC_quant_type = 0
    info.iommu_domain_id = iommu_domain_id
    info.group_size = group_size
    return info


def _set_dims(attr, dims, tensor_type):
    attr.n_dims = len(dims)
    for i, dim in enumerate(dims):
        attr.dims[i] = dim
    attr.size = 0
    attr.type = tensor_type


def attr_summary(attr):
    dims = [int(attr.dims[i]) for i in range(int(attr.n_dims))]
    return {
        "name": attr.name.split(b"\0", 1)[0].decode("ascii", "ignore"),
        "dims": dims,
        "size": int(attr.size),
        "type": int(attr.type),
    }


class MatmulSession:
    def __init__(self, m, k, n, matmul_type, b_layout=0, ac_layout=0, core_mask=RKNN_NPU_CORE_0):
        self.info = make_info(m, k, n, matmul_type, b_layout=b_layout, ac_layout=ac_layout)
        self.io_attr = rknn_matmul_io_attr()
        self.ctx = rknn_matmul_ctx()
        ret = lib.rknn_matmul_create(ctypes.byref(self.ctx), ctypes.byref(self.info), ctypes.byref(self.io_attr))
        if ret < 0:
            raise RuntimeError(f"rknn_matmul_create failed: {ret}")
        ret = lib.rknn_matmul_set_core_mask(self.ctx, core_mask)
        if ret < 0:
            raise RuntimeError(f"rknn_matmul_set_core_mask failed: {ret}")
        self._mems = []

    def alloc(self, size, use_create_mem2=False):
        if use_create_mem2:
            mem = lib.rknn_create_mem2(self.ctx, size, 0)
        else:
            mem = lib.rknn_create_mem(self.ctx, size)
        if not mem:
            raise RuntimeError(f"rknn_create_mem failed size={size}")
        self._mems.append(mem)
        return mem

    def set_io_mem(self, mem, attr):
        ret = lib.rknn_matmul_set_io_mem(self.ctx, mem, ctypes.byref(attr))
        if ret < 0:
            raise RuntimeError(f"rknn_matmul_set_io_mem failed: {ret}")

    def sync(self, mem, mode):
        ret = lib.rknn_mem_sync(self.ctx, mem, mode)
        if ret < 0:
            raise RuntimeError(f"rknn_mem_sync failed: {ret}")

    def run(self):
        ret = lib.rknn_matmul_run(self.ctx)
        if ret < 0:
            raise RuntimeError(f"rknn_matmul_run failed: {ret}")

    def close(self):
        for mem in reversed(self._mems):
            lib.rknn_destroy_mem(self.ctx, mem)
        self._mems.clear()
        lib.rknn_matmul_destroy(self.ctx)
