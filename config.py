import numpy as np
import torch

# config 1: define compression method (DEPRECATED)
# compression_method = ["no_compress", "rle", "self_quant", "baseline"][0]  # define compression method here by change index
# compression_mapping = {
#     "no_compress": (no_compress, no_compress),
#     "rle": (rle_compress, rle_decompress),
#     "self_quant": (quantize_lossy_compress, quantize_lossy_decompress),
#     "baseline": (baseline_quantize, baseline_dequantize),  # convert float32 to float16 and vice versa
# }
#
# compress, decompress = compression_mapping[compression_method]


# config 2: debug mode
DEBUG = 1


# config 3: model

# config 4: dataset


# config 5: transport protocal
protocol = ["TCP", "MLT"][1]

# config 6: bounded-loss tolerance
loss_tolerance = 0
CHUNK_SIZE = 8192 # Size of each chunk to send over the network, in bytes

# config 7: Mappings for Dtypes
# These mappings help convert between torch.dtype, its string representation,
# and the corresponding numpy.dtype needed for some operations.

TORCH_DTYPE_TO_STR = {
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.float16: "torch.float16",
    torch.complex32: "torch.complex32",  # Typically 2x float16, needs special handling
    torch.complex64: "torch.complex64",  # Typically 2x float32
    torch.complex128: "torch.complex128",  # Typically 2x float64
    torch.int8: "torch.int8",
    torch.int16: "torch.int16",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.uint8: "torch.uint8",  # Note: PyTorch doesn't have other unsigned int types like uint16/32/64
    torch.bool: "torch.bool",
}
STR_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_STR.items()}

# Mapping to NumPy dtypes for using np.frombuffer
TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    # torch.bfloat16: would need custom handling, not directly supported by np.frombuffer easily
    torch.complex64: np.complex64,  # (torch complex64 is np complex64, i.e., 2x float32)
    torch.complex128: np.complex128,  # (torch complex128 is np complex128, i.e., 2x float64)
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,  # Note: NumPy's boolean type is np.bool_
}

STR_TO_NUMPY_DTYPE = {str(k): v for k, v in TORCH_TO_NUMPY_DTYPE.items()}

# config 8: drop rate
BEGIN_DROP = 0.0
MID_DROP = 0.0
FINAL_DROP = 0.0

# config 9: tcp max retries
TCP_MAX_RETRIES = 3  # Maximum number of retries for TCP connections

# config 10: timeout
probe_response_timeout = 0.001  # Timeout for probe responses in seconds

# config 11: UDP send rate
UDP_RATE = 10  # Mbps