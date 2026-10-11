"""Imports GPU packages only when inference or training is requested."""
import importlib.metadata
import os
import random
import sys


def reference_backend():
    """Explicit knobs validated by the SSD GPU diagnostic; tensor dtype is FP32."""
    return {"cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_enabled": True,
            "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest", "deterministic_algorithms": False}


def backend_state():
    import torch
    return {"cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "default_dtype": str(torch.get_default_dtype()),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def configure_reference_backend(torch):
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True


def environment():
    result = {"python": sys.version, "packages": {}}
    for name in ["torch", "torchvision", "ultralytics", "transformers", "timm",
                 "numpy", "scipy", "Pillow", "matplotlib", "safetensors", "PyYAML"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    return result


def prepare_device(requested="auto", seed=42, training=False):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    available = torch.cuda.is_available()
    if requested == "auto":
        requested = "cuda:0" if available else "cpu"
    if str(requested).startswith("cuda") and not available:
        raise RuntimeError("CUDA недоступна. Проверь окружение cv-project.")
    if training and available and str(requested) == "cpu":
        raise ValueError("При доступном GPU обучение должно использовать CUDA.")
    if available:
        torch.cuda.manual_seed_all(seed)
    configure_reference_backend(torch)
    return torch.device(requested)


def ensure_finite(model):
    import torch
    bad = [name for name, value in model.state_dict().items()
           if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
           and not torch.isfinite(value).all().item()]
    if bad:
        raise ValueError("NaN/Inf в весах: " + ", ".join(bad[:8]))
