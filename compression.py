import torch
from typing import Any, Dict, List, Tuple, Set


def no_compress(data):
    # a placeholder function for no compression
    return data

def rle_compress(data: Dict[str, torch.Tensor]) -> Dict[str, Tuple[List[Tuple[float, int]], Any]]:
    """Compress gradients using Run-Length Encoding (RLE)."""
    # {fc.weight: [(-0.1, 3), (0.2, 2), ...], fc.bias: [(0.0, 5), ...]}
    compressed = {}
    for name, tensor in data.items():
        tensor_flat = tensor.flatten().tolist()  # Flatten and convert to list
        compressed_runs = []
        current_value = tensor_flat[0]
        count = 1
        for value in tensor_flat[1:]:
            if value == current_value:
                count += 1
            else:
                compressed_runs.append((current_value, count))
                current_value = value
                count = 1
        compressed_runs.append((current_value, count))  # Add the last run
        # Store the compressed runs and the original shape
        compressed[name] = (compressed_runs, tensor.shape)
    return compressed


def rle_decompress(
    compressed: Dict[str, Tuple[List[Tuple[float, int]], Any]]
) -> Dict[str, torch.Tensor]:
    """Decompress gradients using Run-Length Encoding (RLE)."""
    decompressed = {}
    for name, (runs, original_shape) in compressed.items():
        tensor_flat = []
        for value, count in runs:
            tensor_flat.extend([value] * count)
        # Reshape to the original shape
        decompressed[name] = torch.tensor(tensor_flat).reshape(original_shape)
    return decompressed


def quantize_lossy_compress(gradients: dict, num_bits=8):
    """
    Quantizes the gradient tensors using uniform quantization (loseless approach).

    Parameters:
    gradients (dict): Dictionary of named PyTorch gradient tensors.
    num_bits (int): Number of bits for quantization (default: 8-bit).

    Returns:
    dict: Dictionary containing quantized values and metadata for reconstruction.
    """
    quantized_data = {}
    levels = 2**num_bits - 1

    for name, grad in gradients.items():
        min_val, max_val = grad.min(), grad.max()
        quantized_values = torch.round(
            (grad - min_val) / (max_val - min_val) * levels
        ).to(torch.uint8)

        quantized_data[name] = {
            "min_val": min_val.item(),
            "max_val": max_val.item(),
            "quantized_values": quantized_values.cpu().numpy().tolist(),
            "num_bits": num_bits,
        }

    return quantized_data


def quantize_lossy_decompress(q_data: dict):
    """
    Decompresses the quantized gradient data back to PyTorch tensors.

    Parameters:
    q_data (dict): Dictionary of quantized gradient data from quantize_gradients().

    Returns:
    dict: Dictionary of decompressed PyTorch tensors.
    """
    decompressed_data = {}

    for name, data in q_data.items():
        levels = 2 ** data["num_bits"] - 1
        quantized_values = torch.tensor(data["quantized_values"], dtype=torch.float32)
        decompressed_values = data["min_val"] + (quantized_values / levels) * (
            data["max_val"] - data["min_val"]
        )
        decompressed_data[name] = decompressed_values

    return decompressed_data

# this approach won't work because all the gradients are in the range of (-1,1)
# which all will be quantized to 0
def integerize_lossy_compress(gradients: dict):
    """
    Converts all gradient values to integers by zeroing out the floating-point precision.

    Parameters:
    gradients (dict): Dictionary of named PyTorch gradient tensors.

    Returns:
    dict: Dictionary containing integerized gradient tensors.
    """
    integerized_data = {}

    for name, grad in gradients.items():
        integerized_values = torch.round(grad).to(torch.int32)
        integerized_data[name] = integerized_values

    return integerized_data

def baseline_quantize(gradients: dict, type: torch.dtype=torch.float16):
    """
    Quantizes the gradient tensors using torch.to() method.

    Parameters:
    gradients (dict): Dictionary of named PyTorch gradient tensors.

    Returns:
    dict: Dictionary containing quantized values and metadata for reconstruction.
    """
    quantized_data = {}

    for name, grad in gradients.items():
        quantized_values = grad.to(type)
        quantized_data[name] = quantized_values

    return quantized_data

def baseline_dequantize(q_data: dict):
    """
    Dequantizes the quantized gradient tensors using torch.to() method.

    Parameters:
    q_data (dict): Dictionary of quantized gradient tensors.

    Returns:
    dict: Dictionary containing dequantized PyTorch tensors.
    """
    dequantized_data = {}

    for name, grad in q_data.items():
        dequantized_values = grad.to(torch.float32)
        dequantized_data[name] = dequantized_values

    return dequantized_data


def print_gradients(
    gradients: dict,
    host: str,
    gradient_type: str = "Original",
    worker_id: int = 0,
):
    """
    Helper function: Prints the gradients for debugging purposes.
    """
    for name, grad in gradients.items():
        if grad is not None:
            print(f"{gradient_type} Gradient for {name} on {host} {worker_id}:")
            for idx, value in enumerate(grad.flatten()):
                if idx % 1000 == 0:
                    print(f"    [{idx}]: {value}")
