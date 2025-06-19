import torch

def tensors_to_lists(tensor_dict: dict) -> dict:
    """Converts a dictionary of tensors to a dictionary of lists for JSON serialization."""
    if not isinstance(tensor_dict, dict):
        return {}
    # Handles tensors and also keeps non-tensor values (like epoch number) as is.
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v
        for k, v in tensor_dict.items()
    }
