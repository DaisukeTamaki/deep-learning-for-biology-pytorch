import numpy as np
import torch


def dna_to_one_hot(dna_sequence: str) -> np.ndarray:
  """Convert DNA into a one-hot encoded format with channel ordering ACGT."""
  base_to_one_hot = {
    "A": (1, 0, 0, 0),
    "C": (0, 1, 0, 0),
    "G": (0, 0, 1, 0),
    "T": (0, 0, 0, 1),
    "N": (1, 1, 1, 1),
  }
  one_hot_encoded = np.array([base_to_one_hot[base] for base in dna_sequence])
  return one_hot_encoded


def one_hot_to_dna(one_hot_encoded: np.ndarray) -> str:
  """Convert one-hot encoded array back to DNA sequence."""
  one_hot_to_base = {
    (1, 0, 0, 0): "A",
    (0, 1, 0, 0): "C",
    (0, 0, 1, 0): "G",
    (0, 0, 0, 1): "T",
    (1, 1, 1, 1): "N",
  }

  dna_sequence = "".join(
    one_hot_to_base[tuple(base)] for base in one_hot_encoded
  )
  return dna_sequence


def compute_input_gradient(model, sequence, device=None):
  """Compute input gradient for a one-hot DNA sequence."""
  if len(sequence.shape) != 2:
    raise ValueError("Input must be a single one-hot encoded DNA sequence.")

  if device is None:
    device = next(model.parameters()).device

  x = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
  x.requires_grad_(True)

  model.eval()
  output = model(x).mean()
  output.backward()

  gradient = x.grad.squeeze(0).detach().cpu().numpy()
  return gradient


def filter_sequences_by_label(dataset, target_label, max_count):
  """Filter up to N sequences matching a target label."""
  sequences = []
  for i, label in enumerate(dataset["labels"]):
    if label == target_label:
      sequences.append(dataset["sequences"][i])
      if len(sequences) == max_count:
        break
  return sequences
