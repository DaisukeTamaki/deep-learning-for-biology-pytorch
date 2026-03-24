from typing import Callable

import torch
from tqdm import tqdm

from dlfb_pytorch.graphs.dataset import Dataset
from dlfb_pytorch.graphs.model import DdiModel
from dlfb_pytorch.utils.metrics_logger import MetricsLogger
from dlfb_pytorch.utils.restore import restorable


@restorable
def train(
  model: DdiModel,
  optimizer: torch.optim.Optimizer,
  dataset_splits: dict[str, Dataset],
  num_epochs: int,
  loss_fn: Callable,
  norm_loss: bool = False,
  eval_every: int = 10,
) -> tuple[DdiModel, torch.optim.Optimizer, dict[str, dict[str, list[dict[str, float]]]]]:
  """Training loop for the drug-drug interaction model."""
  device = next(model.parameters()).device
  metrics = MetricsLogger()
  batch_size = optimal_batch_size(dataset_splits)

  epochs = tqdm(range(num_epochs))
  for epoch in epochs:
    epochs.set_description(f"Epoch {epoch + 1}")

    model.train()
    for pairs_batch in dataset_splits["train"].pairs.get_train_batches(
      batch_size
    ):
      pairs_batch = {k: v.to(device) for k, v in pairs_batch.items()}
      batch_metrics = train_step(
        model,
        optimizer,
        dataset_splits["train"].graph,
        pairs_batch,
        loss_fn,
        norm_loss,
      )
      metrics.log_step(split="train", **batch_metrics)

    if epoch % eval_every == 0:
      model.eval()
      with torch.no_grad():
        for pairs_batch in dataset_splits["valid"].pairs.get_eval_batches(
          batch_size
        ):
          pairs_batch = {k: v.to(device) for k, v in pairs_batch.items()}
          batch_metrics = eval_step(
            model,
            dataset_splits["valid"].graph,
            pairs_batch,
            loss_fn,
            norm_loss,
          )
          metrics.log_step(split="valid", **batch_metrics)

    metrics.flush(epoch=epoch)
    epochs.set_postfix_str(metrics.latest(["hits@20"]))

  return model, optimizer, metrics.export()


def optimal_batch_size(
  dataset_splits: dict[str, Dataset], remainder_tolerance: float = 0.125
) -> int:
  """Calculates optimal batch size to minimize remainder waste."""
  lengths = [
    min(dataset.pairs.pos.shape[0], dataset.pairs.neg.shape[0])
    for dataset in dataset_splits.values()
  ]

  remainder_thresholds = [
    int(length * remainder_tolerance) for length in lengths
  ]
  max_possible_batch_size = min(lengths)

  for batch_size in range(max_possible_batch_size, 0, -1):
    remainders = [length % batch_size for length in lengths]
    if all(
      remainder <= threshold
      for remainder, threshold in zip(remainders, remainder_thresholds)
    ):
      return batch_size
  return max_possible_batch_size


def binary_log_loss(scores: dict[str, torch.Tensor]) -> torch.Tensor:
  """Computes the binary log loss for positive and negative drug pairs."""
  probs_pos = torch.sigmoid(scores["pos"]).clamp(1e-7, 1 - 1e-7)
  probs_neg = torch.sigmoid(scores["neg"]).clamp(1e-7, 1 - 1e-7)

  pos_loss = -torch.log(probs_pos).mean()
  neg_loss = -torch.log(1 - probs_neg).mean()

  return pos_loss + neg_loss


def auc_loss(scores: dict[str, torch.Tensor]) -> torch.Tensor:
  """Computes AUC-based loss for positive and negative drug pairs."""
  return torch.square(1 - (scores["pos"] - scores["neg"])).sum()


def train_step(
  model: DdiModel,
  optimizer: torch.optim.Optimizer,
  graph,
  pairs: dict[str, torch.Tensor],
  loss_fn: Callable = binary_log_loss,
  norm_loss: bool = False,
) -> dict[str, float]:
  """Performs a single training step, updating model parameters."""
  optimizer.zero_grad()
  scores = model(graph, pairs)
  loss = loss_fn(scores)
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    metric = evaluate_hits_at_20(scores)

  metrics = {"loss": loss.item(), "hits@20": metric}
  if norm_loss:
    metrics["loss"] = metrics["loss"] / (
      pairs["pos"].shape[0] + pairs["neg"].shape[0]
    )

  return metrics


@torch.no_grad()
def eval_step(
  model: DdiModel,
  graph,
  pairs: dict[str, torch.Tensor],
  loss_fn: Callable = binary_log_loss,
  norm_loss: bool = False,
) -> dict[str, float]:
  """Performs an evaluation step, computing loss and hits@20 metric."""
  scores = model(graph, pairs)
  loss = loss_fn(scores)
  metrics = {"loss": loss.item(), "hits@20": evaluate_hits_at_20(scores)}
  if norm_loss:
    metrics["loss"] = metrics["loss"] / (
      pairs["pos"].shape[0] + pairs["neg"].shape[0]
    )
  return metrics


def evaluate_hits_at_20(scores: dict[str, torch.Tensor]) -> float:
  """Computes the hits@20 metric capturing positive pairs ranking."""
  neg_sorted = torch.sort(scores["neg"].detach()).values
  kth_score_in_negative_edges = neg_sorted[-20]
  return (
    (scores["pos"].detach() > kth_score_in_negative_edges).float().sum()
    / scores["pos"].shape[0]
  ).item()
