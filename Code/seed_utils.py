# seed_utils.py
import os, random, numpy as np

def set_all_seeds(seed: int, deterministic_torch: bool = True):
    """
    Best-effort reproducibility across Python, NumPy, TensorFlow, and PyTorch.
    Determinism on GPU is not perfect across all ops, but this hits the big levers.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")   
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
