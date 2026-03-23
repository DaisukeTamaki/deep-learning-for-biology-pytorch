import torch


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
  """Compute accuracy as the proportion of correct predictions."""
  return torch.mean((y_true == y_pred).float())


def calculate_true_positives(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Count true positives for a given class label."""
  return torch.sum((y_pred == label) & (y_true == label))


def calculate_false_positives(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Count false positives for a given class label."""
  return torch.sum((y_pred == label) & (y_true != label))


def calculate_false_negatives(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Count false negatives for a given class label."""
  return torch.sum((y_pred != label) & (y_true == label))


def calculate_true_negatives(
  y_true: torch.Tensor, y_pred: torch.Tensor, label: int
) -> torch.Tensor:
  """Count true negatives by excluding the given class label."""
  return torch.sum((y_pred != label) & (y_true != label))
