import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dlfb_pytorch.dna.utils import dna_to_one_hot


def load_dataset_splits(
  path, transcription_factor, batch_size: int | None = None
):
  """Load dataset splits (train, valid, test) as DataLoaders."""
  dataset_splits = {}
  for split in ["train", "valid", "test"]:
    dataset = load_dataset(
      sequence_db=f"{path}/{transcription_factor}_{split}_sequences.csv"
    )
    dl = create_dataloader(dataset, batch_size, is_training=(split == "train"))
    dataset_splits[split] = dl
  return dataset_splits


def create_dataloader(
  dataset, batch_size: int | None = None, is_training: bool = False
):
  """Convert DNA sequences and labels to a PyTorch DataLoader."""
  sequences = torch.tensor(dataset["sequences"], dtype=torch.float32)
  labels = torch.tensor(dataset["labels"], dtype=torch.float32)
  ds = TensorDataset(sequences, labels)
  batch_size = batch_size or len(dataset["labels"])
  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=is_training,
    drop_last=is_training,
  )


def load_dataset(sequence_db) -> dict[str, np.ndarray]:
  """Load sequences and labels from a CSV into numpy arrays."""
  df = pd.read_csv(sequence_db)
  return {
    "labels": df["label"].to_numpy()[:, None],
    "sequences": np.array([dna_to_one_hot(seq) for seq in df["sequence"]]),
  }
