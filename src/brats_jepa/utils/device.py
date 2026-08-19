import torch


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Resolves execution hardware device.
    Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)
