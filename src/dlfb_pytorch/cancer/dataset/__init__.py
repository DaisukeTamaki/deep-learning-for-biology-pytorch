from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch


@dataclass
class Images:
  """Stores image data and size information."""

  loaded: dict[str, np.memmap]
  size: tuple[int, int, int]


@dataclass
class Dataset:
  """Dataset class storing images and corresponding metadata."""

  metadata: pd.DataFrame
  images: Images
  num_classes: int

  def get_dummy_input(self) -> torch.Tensor:
    """Returns dummy input with the correct shape for the model."""
    return torch.empty((1,) + self.images.size)

  def num_samples(self) -> int:
    """Returns the number of samples in the dataset."""
    return self.metadata.shape[0]

  def get_images(self, preprocessor: Callable, indices) -> np.ndarray:
    """Returns preprocessed images for the given indices."""
    return self.images.loaded[preprocessor.__name__][indices]

  def get_labels(self, indices) -> np.ndarray:
    """Returns integer class labels for the given indices."""
    return self.metadata.loc[indices]["label"].values.astype(np.int64)
