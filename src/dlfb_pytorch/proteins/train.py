import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  dataset_splits: dict[str, DataLoader],
  num_steps: int = 300,
  eval_every: int = 30,
  device: torch.device | None = None,
):
  """Train model using DataLoaders and track performance metrics."""
  if device is None:
    device = next(model.parameters()).device

  train_metrics, valid_metrics = [], []

  train_iter = _repeating_iterator(dataset_splits["train"])

  steps = tqdm(range(num_steps))
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    embeddings, targets = next(train_iter)
    loss = train_step(model, optimizer, embeddings, targets, device)
    train_metrics.append({"step": step, "loss": loss})

    if step % eval_every == 0:
      eval_metrics_list = []
      for embeddings, targets in dataset_splits["valid"]:
        eval_metrics_list.append(
          eval_step(model, embeddings, targets, device)
        )
      valid_metrics.append(
        {"step": step, **pd.DataFrame(eval_metrics_list).mean(axis=0).to_dict()}
      )

  return model, optimizer, {"train": train_metrics, "valid": valid_metrics}


def train_step(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  embeddings: torch.Tensor,
  targets: torch.Tensor,
  device: torch.device,
) -> float:
  """Run a single training step and update model parameters."""
  model.train()
  embeddings, targets = embeddings.to(device), targets.to(device)

  optimizer.zero_grad()
  logits = model(embeddings)
  loss = F.binary_cross_entropy_with_logits(logits, targets)
  loss.backward()
  optimizer.step()

  return loss.item()


@torch.no_grad()
def eval_step(
  model: torch.nn.Module,
  embeddings: torch.Tensor,
  targets: torch.Tensor,
  device: torch.device,
) -> dict[str, float]:
  """Run evaluation step and return mean metrics over targets."""
  model.eval()
  embeddings, targets = embeddings.to(device), targets.to(device)

  logits = model(embeddings)
  loss = F.binary_cross_entropy_with_logits(logits, targets)
  target_metrics = calculate_per_target_metrics(
    logits.cpu().numpy(), targets.cpu().numpy()
  )
  return {
    "loss": loss.item(),
    **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),
  }


def calculate_per_target_metrics(logits, targets):
  """Compute metrics for each target in a multi-label batch."""
  probs = torch.sigmoid(torch.tensor(logits)).numpy()
  target_metrics = []
  for target, prob in zip(targets, probs):
    target_metrics.append(compute_metrics(target, prob))
  return target_metrics


def compute_metrics(
  targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> dict[str, float]:
  """Compute accuracy, recall, precision, auPRC, and auROC."""
  if np.sum(targets) == 0:
    return {
      m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]
    }
  return {
    "accuracy": metrics.accuracy_score(targets, probs >= thresh),
    "recall": metrics.recall_score(targets, probs >= thresh).item(),
    "precision": metrics.precision_score(
      targets,
      probs >= thresh,
      zero_division=0.0,
    ).item(),
    "auprc": metrics.average_precision_score(targets, probs).item(),
    "auroc": metrics.roc_auc_score(targets, probs).item(),
  }


def _repeating_iterator(dataloader: DataLoader):
  """Infinitely cycle through a DataLoader."""
  while True:
    yield from dataloader
