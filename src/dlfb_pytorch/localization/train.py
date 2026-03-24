import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dlfb_pytorch.localization.dataset import Dataset
from dlfb_pytorch.localization.model import LocalizationModel
from dlfb_pytorch.metrics import accuracy_score
from dlfb_pytorch.utils.metrics_logger import MetricsLogger
from dlfb_pytorch.utils.restore import restorable


@restorable
def train(
  model: LocalizationModel,
  optimizer: torch.optim.Optimizer,
  dataset_splits: dict[str, Dataset],
  num_epochs: int,
  batch_size: int,
  classification_weight: float,
  eval_every: int = 10,
) -> tuple[LocalizationModel, torch.optim.Optimizer, dict]:
  """Train the VQ-VAE model with optional classification."""
  device = next(model.parameters()).device
  metrics = MetricsLogger()

  epochs = tqdm(range(num_epochs))
  for epoch in epochs:
    epochs.set_description(f"Epoch {epoch + 1}")

    model.train()
    for batch in dataset_splits["train"].get_batches(batch_size=batch_size):
      batch_metrics = train_step(
        model, optimizer, batch, classification_weight, device
      )
      metrics.log_step(split="train", **batch_metrics)

    if epoch % eval_every == 0:
      model.eval()
      with torch.no_grad():
        for batch in dataset_splits["valid"].get_batches(batch_size=batch_size):
          batch_metrics = eval_step(
            model, batch, classification_weight, device
          )
          metrics.log_step(split="valid", **batch_metrics)

    metrics.flush(epoch=epoch)
    epochs.set_postfix_str(metrics.latest(["total_loss"]))

  return model, optimizer, metrics.export()


def train_step(
  model: LocalizationModel,
  optimizer: torch.optim.Optimizer,
  batch: dict,
  classification_weight: float,
  device: torch.device,
) -> dict[str, float]:
  """Train for a single step."""
  images = (
    torch.from_numpy(np.ascontiguousarray(batch["images"]))
    .permute(0, 3, 1, 2)
    .float()
    .to(device)
  )
  labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

  optimizer.zero_grad()
  x_recon, perplexity, codebook_loss, commitment_loss, logits = model(
    images, is_training=True
  )

  recon_loss = ((x_recon - images) ** 2).mean()
  classification_loss = classification_weight * F.cross_entropy(logits, labels)
  total_loss = recon_loss + codebook_loss + commitment_loss + classification_loss

  total_loss.backward()
  optimizer.step()

  with torch.no_grad():
    acc = accuracy_score(labels, logits.argmax(-1))

  return {
    "total_loss": total_loss.item(),
    "recon_loss": recon_loss.item(),
    "codebook_loss": codebook_loss.item(),
    "commitment_loss": commitment_loss.item(),
    "classification_loss": classification_loss.item(),
    "perplexity": perplexity.item(),
    "accuracy": acc.item(),
  }


@torch.no_grad()
def eval_step(
  model: LocalizationModel,
  batch: dict,
  classification_weight: float,
  device: torch.device,
) -> dict[str, float]:
  """Evaluate on a single batch."""
  images = (
    torch.from_numpy(np.ascontiguousarray(batch["images"]))
    .permute(0, 3, 1, 2)
    .float()
    .to(device)
  )
  labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

  x_recon, perplexity, codebook_loss, commitment_loss, logits = model(
    images, is_training=False
  )

  recon_loss = (0.5 * (x_recon - images) ** 2).mean()
  classification_loss = classification_weight * F.cross_entropy(logits, labels)
  total_loss = recon_loss + codebook_loss + commitment_loss + classification_loss

  acc = accuracy_score(labels, logits.argmax(-1))

  return {
    "total_loss": total_loss.item(),
    "recon_loss": recon_loss.item(),
    "codebook_loss": codebook_loss.item(),
    "commitment_loss": commitment_loss.item(),
    "classification_loss": classification_loss.item(),
    "perplexity": perplexity.item(),
    "accuracy": acc.item(),
  }
