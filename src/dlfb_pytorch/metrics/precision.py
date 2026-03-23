from typing import Literal

import torch

from dlfb_pytorch.metrics import (
  calculate_false_positives,
  calculate_true_positives,
)


def precision_score(
  y_true: torch.Tensor,
  y_preds: torch.Tensor,
  n_labels: int,
  average: Literal["macro", "micro", "weighted"],
) -> torch.Tensor:
  """Computes precision score with macro, micro, or weighted averaging."""
  labels = torch.arange(n_labels, device=y_true.device)
  precisions = torch.stack(
    [calculate_precision_per_class(y_true, y_preds, lb.item()) for lb in labels]
  )

  match average:
    case "macro":
      precision = torch.mean(precisions)
    case "micro":
      tp = torch.sum(y_preds == y_true)
      fp = torch.sum(y_preds != y_true)
      precision = tp / (tp + fp + 1e-8)
    case "weighted":
      support = torch.stack(
        [torch.sum(y_true == lb) for lb in labels]
      ).float()
      precision = torch.sum(precisions * (support / (torch.sum(support) + 1e-8)))
    case _:
      raise ValueError(f"Unsupported average type: {average}")

  return precision


def calculate_precision_per_class(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Computes precision for a single class."""
  tp = calculate_true_positives(y_true, y_pred, label)
  fp = calculate_false_positives(y_true, y_pred, label)
  return tp / (tp + fp + 1e-8)
