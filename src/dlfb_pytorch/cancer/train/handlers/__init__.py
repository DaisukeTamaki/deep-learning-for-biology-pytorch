from typing import Callable

import numpy as np

from dlfb_pytorch.cancer.dataset import Dataset
from dlfb_pytorch.cancer.dataset.preprocessors import crop
from dlfb_pytorch.cancer.train.handlers.samplers import epoch_sampler


class BatchHandler:
  """Helper to provide appropriately prepared batches from a dataset."""

  def __init__(
    self,
    preprocessor: Callable = crop,
    sampler: Callable = epoch_sampler,
    augmentor: Callable | None = None,
  ):
    self.preprocessor = preprocessor
    self.sampler = sampler
    self.augmentor = augmentor

  def get_batches(self, dataset: Dataset, batch_size: int):
    """Prepare dataset batches with the requested image manipulations."""
    self._validate_batch_size(dataset, batch_size)

    for batch_indices in self.sampler(dataset.metadata, batch_size):
      images = dataset.get_images(self.preprocessor, batch_indices)

      if self.augmentor is not None:
        images = np.stack([self.augmentor(img) for img in images])

      batch = {
        "frame_ids": batch_indices,
        "images": images,
        "labels": dataset.get_labels(batch_indices),
      }
      yield batch

  @staticmethod
  def _validate_batch_size(dataset: Dataset, batch_size: int) -> None:
    """Ensures that batch_size is within feasible bounds."""
    if batch_size > dataset.num_samples():
      raise ValueError(
        f"Batch size ({batch_size}) cannot be larger than dataset size "
        f"({dataset.num_samples()})."
      )
    if batch_size < dataset.num_classes:
      raise ValueError(
        f"batch_size ({batch_size}) has to be larger than "
        f"number of unique labels."
      )
