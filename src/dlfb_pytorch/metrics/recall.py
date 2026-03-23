from typing import Literal

import torch

from dlfb_pytorch.metrics import (
  calculate_false_negatives,
  calculate_true_positives,
)


def recall_score(
  y_true: torch.Tensor,
  y_preds: torch.Tensor,
  n_labels: int,
  average: Literal["macro", "micro", "weighted"],
) -> torch.Tensor:
  """Computes recall score with macro, micro, or weighted averaging."""
  labels = torch.arange(n_labels, device=y_true.device)
  recalls = torch.stack(
    [calculate_recall_per_class(y_true, y_preds, lb.item()) for lb in labels]
  )

  match average:
    case "macro":
      recall = torch.mean(recalls)
    case "micro":
      tp = torch.sum(y_preds == y_true)
      fn = torch.sum(y_preds != y_true)
      recall = tp / (tp + fn + 1e-8)
    case "weighted":
      support = torch.stack(
        [torch.sum(y_true == lb) for lb in labels]
      ).float()
      recall = torch.sum(recalls * (support / (torch.sum(support) + 1e-8)))
    case _:
      raise ValueError(f"Unsupported average type: {average}")

  return recall


def calculate_recall_per_class(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Computes recall for a single class."""
  tp = calculate_true_positives(y_true, y_pred, label)
  fn = calculate_false_negatives(y_true, y_pred, label)
  return tp / (tp + fn + 1e-8)
