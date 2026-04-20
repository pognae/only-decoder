import torch


def resolve_runtime_device(device: str) -> str:
    if device == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    if device == "gpu":
        raise RuntimeError("gpu requested but no CUDA/MPS device is available")

    return "cpu"
