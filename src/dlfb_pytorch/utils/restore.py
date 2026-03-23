import json
from functools import wraps
from pathlib import Path

import torch

from dlfb_pytorch.log import log


def restore(store_path, model, optimizer=None):
  """Restore model, optimizer, and metrics from a checkpoint directory."""
  checkpoint_dir = Path(store_path).resolve()
  state = {"model": model, "optimizer": optimizer}
  state, metrics = restore_checkpoint(checkpoint_dir, state)
  return state, metrics


def store(store_path, model, optimizer, metrics) -> None:
  """Save model, optimizer, and metrics to a checkpoint directory."""
  checkpoint_dir = Path(store_path).resolve()
  state = {"model": model, "optimizer": optimizer}
  save_final_checkpoint(checkpoint_dir, state, metrics)


def restorable(train_fn):
  @wraps(train_fn)
  def wrapper(model, optimizer, store_path: str | None = None, **kwargs):
    if store_path:
      checkpoint_dir = Path(store_path).resolve()
      try:
        state = {"model": model, "optimizer": optimizer}
        state, metrics = restore_checkpoint(checkpoint_dir, state)
        model, optimizer = state["model"], state["optimizer"]
      except FileNotFoundError:
        log.debug("Training new model, checkpointing state and metrics...")
        model, optimizer, metrics = train_fn(model, optimizer, **kwargs)
        state = {"model": model, "optimizer": optimizer}
        save_final_checkpoint(checkpoint_dir, state, metrics)
    else:
      log.debug("Training new model without checkpointing...")
      model, optimizer, metrics = train_fn(model, optimizer, **kwargs)
    return model, optimizer, metrics

  return wrapper


def restore_checkpoint(checkpoint_dir: Path, state):
  checkpoint_path = checkpoint_dir / "checkpoint.pt"
  metrics_path = checkpoint_dir / "metrics.json"
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

  if state["model"] is not None:
    state["model"].load_state_dict(checkpoint["model_state_dict"])
  if state["optimizer"] is not None and "optimizer_state_dict" in checkpoint:
    state["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])

  metrics = {}
  if metrics_path.exists():
    with open(metrics_path) as f:
      metrics = json.load(f)

  log.debug("Model state and metrics restored from checkpoint")
  return state, metrics


def save_final_checkpoint(checkpoint_dir: Path, state, metrics):
  checkpoint_dir.mkdir(parents=True, exist_ok=True)

  checkpoint = {}
  if state["model"] is not None:
    checkpoint["model_state_dict"] = state["model"].state_dict()
  if state["optimizer"] is not None:
    checkpoint["optimizer_state_dict"] = state["optimizer"].state_dict()

  torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

  with open(checkpoint_dir / "metrics.json", "w") as f:
    json.dump(metrics, f)
