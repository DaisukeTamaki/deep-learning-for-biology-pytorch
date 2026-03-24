from typing import Iterator

import numpy as np
import pandas as pd


def epoch_sampler(
  metadata: pd.DataFrame, batch_size: int
) -> Iterator[np.ndarray]:
  """Yields batches of indices from metadata sequentially."""
  frame_ids = metadata["frame_id"].to_numpy()
  shuffled_frame_ids = np.random.permutation(frame_ids)
  for i in range(0, len(shuffled_frame_ids), batch_size):
    yield shuffled_frame_ids[i : i + batch_size]


def repeating_sampler(
  metadata: pd.DataFrame, batch_size: int
) -> Iterator[np.ndarray]:
  """Continuously generates random batches by sampling without replacement."""
  frame_ids = metadata["frame_id"].to_numpy()

  while True:
    batch_indices = np.random.choice(
      frame_ids, size=batch_size, replace=False
    )
    yield batch_indices


def balanced_sampler(
  metadata: pd.DataFrame, batch_size: int
) -> Iterator[np.ndarray]:
  """Yields balanced batches by sampling equal number of instances per class."""
  labels = metadata["label"].unique()
  samples_per_class = (batch_size // len(labels)) + 1

  while True:
    batch_indices = []
    for label in labels:
      frame_ids = metadata.loc[
        metadata["label"] == label, "frame_id"
      ].to_numpy()
      sampled_frame_ids = np.random.choice(
        frame_ids, size=samples_per_class, replace=False
      )
      batch_indices.extend(sampled_frame_ids)

    shuffled_batch_indices = np.random.permutation(batch_indices)
    yield shuffled_batch_indices[:batch_size]
