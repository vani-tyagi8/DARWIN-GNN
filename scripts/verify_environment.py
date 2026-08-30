import json
import platform

import torch
import torch_geometric

report = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(report, indent=2))

