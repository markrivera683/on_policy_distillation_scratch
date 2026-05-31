import torch

import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Any


def logprob(dis: torch.Tensor | None = None) -> torch.Tensor:
    pass