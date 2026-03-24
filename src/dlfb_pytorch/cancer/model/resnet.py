import torch
import torch.nn as nn

from dlfb_pytorch.cancer.model import RESNET_NAMES, SkinLesionClassifierHead
from dlfb_pytorch.cancer.utils import build_param_groups, split_decay_params


class ResNetFromScratch(nn.Module):
  """ResNet model initialized from scratch with a custom classification head."""

  def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
    super().__init__()
    self.backbone = self._create_backbone(layers)
    hidden_size = self.backbone.config.hidden_sizes[-1]
    self.head = SkinLesionClassifierHead(num_classes, dropout_rate, hidden_size)

  def _create_backbone(self, layers):
    from transformers import ResNetConfig, ResNetModel

    config = ResNetConfig.from_pretrained(RESNET_NAMES[layers])
    return ResNetModel(config)

  def forward(self, x):
    """Runs a forward pass through the model. Input: (B, C, H, W)."""
    features = self.backbone(x).pooler_output  # (B, hidden_size)
    features = features.flatten(1)
    return self.head(features)

  def create_optimizer(self, lr: float = 0.001, weight_decay: float = 0.0):
    """Creates an AdamW optimizer with optional weight decay masking."""
    if weight_decay > 0:
      decay, no_decay = split_decay_params(self)
      return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
      ], lr=lr)
    return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0)


class FinetunedResNet(ResNetFromScratch):
  """ResNet model with pretrained weights and full fine-tuning."""

  def _create_backbone(self, layers):
    from transformers import ResNetModel

    return ResNetModel.from_pretrained(RESNET_NAMES[layers])


class FinetunedHeadResNet(FinetunedResNet):
  """ResNet model with a frozen backbone and trainable classification head."""

  def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
    super().__init__(num_classes, layers, dropout_rate)
    for param in self.backbone.parameters():
      param.requires_grad = False

  def create_optimizer(self, lr: float = 0.001, weight_decay: float = 0.0):
    """Creates optimizer for head parameters only (backbone is frozen)."""
    head_groups = build_param_groups(
      self.head.named_parameters(), lr, weight_decay
    )
    return torch.optim.AdamW(head_groups)


class PartiallyFinetunedResNet(FinetunedResNet):
  """ResNet model with selective fine-tuning of deeper layers."""

  def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
    super().__init__(num_classes, layers, dropout_rate)
    for name, param in self.backbone.named_parameters():
      if "stages.3" not in name:
        param.requires_grad = False

  def create_optimizer(self, lr: float = 0.001, weight_decay: float = 0.0):
    """Freezes early layers, fine-tunes stage 3 at reduced LR, head at full LR."""
    head_params = list(self.head.named_parameters())
    stage3_params = [
      (n, p) for n, p in self.backbone.named_parameters() if p.requires_grad
    ]

    groups = build_param_groups(head_params, lr, weight_decay)
    groups.extend(build_param_groups(stage3_params, 1e-5, weight_decay))

    return torch.optim.AdamW(groups)
