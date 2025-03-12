from compression import *

# config 1: define compression method
compression_method = ["no_compress", "rle", "self_quant", "baseline"][0]  # define compression method here by change index
compression_mapping = {
    "no_compress": (no_compress, no_compress),
    "rle": (rle_compress, rle_decompress),
    "self_quant": (quantize_lossy_compress, quantize_lossy_decompress),
    "baseline": (baseline_quantize, baseline_dequantize), # convert float32 to float16 and vice versa
}

compress, decompress = compression_mapping[compression_method]


# config 2: debug mode
DEBUG = 0