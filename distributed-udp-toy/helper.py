import torch
import json

def tensors_to_lists(tensor_dict: dict) -> dict:
    """Converts a dictionary of tensors to a dictionary of lists for JSON serialization."""
    if not isinstance(tensor_dict, dict):
        return {}
    # Handles tensors and also keeps non-tensor values (like epoch number) as is.
    return {
        k: v.tolist() if isinstance(v, torch.Tensor) else v
        for k, v in tensor_dict.items()
    }

def write_to_json(worker_id, data):
    json_filename = f"dummy_gradient_{worker_id}.json"
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"[Worker {worker_id}] Dummy gradients have been exported to {json_filename}")
