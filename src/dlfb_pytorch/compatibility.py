import os


def patch() -> None:
  patch_numpy_nep50_warning()
  patch_ogb_dataset_loader()


def patch_numpy_nep50_warning() -> None:
  import numpy as np

  def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
      return x

    return npwarn_decorator

  np._no_nep50_warning = getattr(
    np, "_no_nep50_warning", dummy_npwarn_decorator_factory
  )


def patch_ogb_dataset_loader() -> None:
  os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "yes"
