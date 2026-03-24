import torch
import torch.nn as nn


class Model(nn.Module):
  """Simple MLP for protein function prediction."""

  def __init__(self, num_targets: int, input_dim: int = 640, dim: int = 256):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, dim * 2),
      nn.GELU(),
      nn.Linear(dim * 2, dim),
      nn.GELU(),
      nn.Linear(dim, num_targets),
    )

  def forward(self, x):
    """Apply MLP layers to input features."""
    return self.net(x)

  def create_optimizer(self, lr: float = 0.001) -> torch.optim.Optimizer:
    """Create an Adam optimizer for this model."""
    return torch.optim.Adam(self.parameters(), lr=lr)
