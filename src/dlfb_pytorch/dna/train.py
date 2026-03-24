from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dlfb_pytorch.utils.metrics_logger import MetricsLogger
from dlfb_pytorch.utils.restore import restorable


@restorable
def train(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  dataset_splits: dict[str, DataLoader],
  num_steps: int,
  eval_every: int = 100,
  device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
  """Train a model and log metrics over steps."""
  if device is None:
    device = next(model.parameters()).device

  metrics = MetricsLogger()
  train_iter = _repeating_iterator(dataset_splits["train"])

  steps = tqdm(range(num_steps))
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    sequences, labels = next(train_iter)
    batch_metrics = train_step(model, optimizer, sequences, labels, device)
    metrics.log_step(split="train", **batch_metrics)

    if step % eval_every == 0:
      for sequences, labels in dataset_splits["valid"]:
        batch_metrics = eval_step(model, sequences, labels, device)
        metrics.log_step(split="valid", **batch_metrics)
      metrics.flush(step=step)

    steps.set_postfix_str(metrics.latest(["loss"]))

  return model, optimizer, metrics.export()


def train_step(model, optimizer, sequences, labels, device):
  """Run a training step and update parameters."""
  model.train()
  sequences, labels = sequences.to(device), labels.to(device)

  optimizer.zero_grad()
  logits = model(sequences)
  loss = F.binary_cross_entropy_with_logits(logits, labels)
  loss.backward()
  optimizer.step()

  return {"loss": loss.item()}


@torch.no_grad()
def eval_step(model, sequences, labels, device):
  """Evaluate model on a single batch."""
  model.eval()
  sequences, labels = sequences.to(device), labels.to(device)

  logits = model(sequences)
  loss = F.binary_cross_entropy_with_logits(logits, labels)
  return {
    "loss": loss.item(),
    **compute_metrics(
      labels.cpu().numpy(), logits.cpu().numpy()
    ),
  }


def compute_metrics(y_true: np.ndarray, logits: np.ndarray):
  """Compute accuracy and auROC for model predictions."""
  probs = torch.sigmoid(torch.tensor(logits)).numpy()
  return {
    "accuracy": accuracy_score(y_true, probs >= 0.5),
    "auc": roc_auc_score(y_true, logits).item(),
  }


def _repeating_iterator(dataloader: DataLoader):
  """Infinitely cycle through a DataLoader."""
  while True:
    yield from dataloader
