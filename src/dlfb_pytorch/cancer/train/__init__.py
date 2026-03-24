from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dlfb_pytorch.cancer.dataset import Dataset
from dlfb_pytorch.cancer.dataset.preprocessors import crop
from dlfb_pytorch.cancer.train.handlers import BatchHandler
from dlfb_pytorch.cancer.train.handlers.samplers import repeating_sampler
from dlfb_pytorch.metrics.precision import precision_score
from dlfb_pytorch.metrics.recall import recall_score
from dlfb_pytorch.utils.metrics_logger import MetricsLogger
from dlfb_pytorch.utils.restore import restorable


@restorable
def train(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  dataset_splits: dict[str, Dataset],
  num_steps: int,
  batch_size: int,
  preprocessor: Callable = crop,
  sampler: Callable = repeating_sampler,
  augmentor: Callable = None,
  eval_every: int = 10,
  scheduler: object = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
  """Trains a model using the provided dataset splits and logs metrics."""
  device = next(model.parameters()).device
  num_classes = dataset_splits["train"].num_classes
  metrics = MetricsLogger()

  train_batcher = BatchHandler(preprocessor, sampler, augmentor)
  train_batches = train_batcher.get_batches(
    dataset_splits["train"], batch_size
  )

  steps = tqdm(range(num_steps))
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    model.train()
    train_batch = next(train_batches)
    batch_metrics = train_step(
      model, optimizer, train_batch, num_classes, device
    )
    metrics.log_step(split="train", **batch_metrics)

    if scheduler is not None:
      scheduler.step()

    if step % eval_every == 0:
      model.eval()
      with torch.no_grad():
        for batch in BatchHandler(preprocessor).get_batches(
          dataset_splits["valid"], batch_size
        ):
          batch_metrics = eval_step(model, batch, num_classes, device)
          metrics.log_step(split="valid", **batch_metrics)
      metrics.flush(step=step)

    steps.set_postfix_str(metrics.latest(["loss"]))

  return model, optimizer, metrics.export()


def train_step(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  batch: dict,
  num_classes: int,
  device: torch.device,
) -> dict[str, float]:
  """Performs a single training step and returns updated metrics."""
  images = (
    torch.from_numpy(np.ascontiguousarray(batch["images"]))
    .permute(0, 3, 1, 2)
    .float()
    .to(device)
  )
  labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

  optimizer.zero_grad()
  logits = model(images)
  loss = F.cross_entropy(logits, labels)
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    batch_metrics = compute_metrics(labels, logits, num_classes)

  return {"loss": loss.item(), **batch_metrics}


@torch.no_grad()
def eval_step(
  model: torch.nn.Module,
  batch: dict,
  num_classes: int,
  device: torch.device,
) -> dict[str, float]:
  """Evaluates model performance on a batch and computes metrics."""
  images = (
    torch.from_numpy(np.ascontiguousarray(batch["images"]))
    .permute(0, 3, 1, 2)
    .float()
    .to(device)
  )
  labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

  logits = model(images)
  loss = F.cross_entropy(logits, labels)

  return {"loss": loss.item(), **compute_metrics(labels, logits, num_classes)}


def compute_metrics(
  y_true: torch.Tensor, logits: torch.Tensor, n_labels: int
) -> dict[str, float]:
  """Computes weighted precision and recall metrics from logits and labels."""
  y_pred = logits.argmax(dim=1)
  return {
    "recall_weighted": recall_score(
      y_true, y_pred, n_labels, average="weighted"
    ).item(),
    "precision_weighted": precision_score(
      y_true, y_pred, n_labels, average="weighted"
    ).item(),
  }


def get_predictions(
  model: torch.nn.Module,
  dataset: Dataset,
  preprocessor: Callable,
  batch_size: int = 32,
) -> pd.DataFrame:
  """Generates predictions for entire dataset using the current model."""
  device = next(model.parameters()).device
  model.eval()
  dfs = []
  with torch.no_grad():
    for batch in BatchHandler(preprocessor).get_batches(dataset, batch_size):
      images = (
        torch.from_numpy(np.ascontiguousarray(batch["images"]))
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
      )
      logits = model(images)
      pred = logits.argmax(dim=-1).cpu().tolist()
      dfs.append(pd.DataFrame({"frame_id": batch["frame_ids"], "pred": pred}))
  predictions = pd.concat(dfs).merge(
    dataset.metadata[["frame_id", "class", "label"]], on="frame_id"
  )
  return predictions
