from typing import Literal

import torch

from dlfb_pytorch.metrics.precision import calculate_precision_per_class
from dlfb_pytorch.metrics.recall import calculate_recall_per_class


def f1_score(
  y_true: torch.Tensor,
  y_pred: torch.Tensor,
  n_labels: int,
  average: Literal["macro", "micro", "weighted"],
) -> torch.Tensor:
  """Compute F1 score with macro, micro, or weighted averaging."""
  labels = torch.arange(n_labels, device=y_true.device)
  f1_scores = torch.stack(
    [calculate_f1_per_class(y_true, y_pred, lb.item()) for lb in labels]
  )

  match average:
    case "macro":
      f1 = torch.mean(f1_scores)
    case "micro":
      tp = torch.sum(y_pred == y_true)
      fn_fp = torch.sum(y_pred != y_true)
      precision_micro = tp / (tp + fn_fp + 1e-8)
      recall_micro = tp / (tp + fn_fp + 1e-8)
      f1 = (
        2
        * (precision_micro * recall_micro)
        / (precision_micro + recall_micro + 1e-8)
      )
    case "weighted":
      support = torch.stack(
        [torch.sum(y_true == lb) for lb in labels]
      ).float()
      f1 = torch.sum(f1_scores * (support / torch.sum(support)))
    case _:
      raise ValueError(
        f"Unsupported average type '{average}'. Choose from 'macro', 'micro',"
        " or 'weighted'."
      )
  return f1


def calculate_f1_per_class(
  y_true: torch.Tensor, y_pred: torch.Tensor, class_label: int
) -> torch.Tensor:
  """Calculate F1 score for a single class."""
  precision = calculate_precision_per_class(y_true, y_pred, class_label)
  recall = calculate_recall_per_class(y_true, y_pred, class_label)
  combined = precision + recall
  return torch.where(
    combined > 0,
    2 * (precision * recall) / (combined + 1e-8),
    torch.tensor(0.0, device=y_true.device),
  )
