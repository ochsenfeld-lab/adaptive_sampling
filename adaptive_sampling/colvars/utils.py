import torch
from typing import Tuple

def _partial_derivative(
    f: torch.tensor, *args: Tuple[torch.tensor]
) -> Tuple[torch.tensor]:
    """get partial derivative of arbitrary function from torch.autograd

    Args:
        f (torch.tensor): function f(*args) to differentiate
        *args (torch.tensor): variables of f for which derivative is computed

    Returns:
        partial_derivatives (Tuple[torch.tensor]): derivatives of f with respect to args
    """
    return torch.autograd.grad(f, *args, allow_unused=True)