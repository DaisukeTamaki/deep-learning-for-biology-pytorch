import torch
import torch.nn as nn
import torch.nn.functional as F

RESNET_NAMES = {
  18: "microsoft/resnet-18",
  34: "microsoft/resnet-34",
  50: "microsoft/resnet-50",
}


class SkinLesionClassifierHead(nn.Module):
  """Skin lesion classification MLP head."""

  def __init__(self, num_classes: int, dropout_rate: float, input_dim: int = 2048):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, num_classes)
    self.dropout = nn.Dropout(dropout_rate)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.xavier_uniform_(self.fc3.weight)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x
