import torch
import torch.nn as nn
import torch.nn.functional as F

from dlfb_pytorch.cancer.model import SkinLesionClassifierHead
from dlfb_pytorch.cancer.utils import split_decay_params


class SimpleCnn(nn.Module):
  """Simple CNN model with small convolutional backbone and classifier head."""

  def __init__(self, num_classes: int, dropout_rate: float = 0.0):
    super().__init__()
    self.backbone = CnnBackbone()
    self.head = SkinLesionClassifierHead(num_classes, dropout_rate, input_dim=256)

  def forward(self, x):
    """Applies the backbone and classifier head to the input."""
    x = self.backbone(x)
    return self.head(x)

  def create_optimizer(self, lr: float = 0.001, weight_decay: float = 0.0):
    """Creates an AdamW optimizer with optional weight decay masking."""
    if weight_decay > 0:
      decay, no_decay = split_decay_params(self)
      return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
      ], lr=lr)
    return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0)


class CnnBackbone(nn.Module):
  """Compact convolutional feature extractor for image data."""

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    with torch.no_grad():
      x = torch.zeros(1, 3, 224, 224)
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      flat_dim = x.numel()

    self.fc = nn.Linear(flat_dim, 256)

  def forward(self, x):
    """Applies two conv-pool blocks and a dense layer to the input."""
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x
