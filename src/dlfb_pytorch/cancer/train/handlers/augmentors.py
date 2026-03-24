import numpy as np
import torch
import torchvision.transforms.functional as TF


def flipping_augmentor(image: np.ndarray) -> np.ndarray:
  """Applies random horizontal and vertical flips."""
  t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
  if torch.rand(1).item() > 0.5:
    t = TF.hflip(t)
  if torch.rand(1).item() > 0.5:
    t = TF.vflip(t)
  return t.permute(1, 2, 0).numpy()


def rich_augmentor(image: np.ndarray) -> np.ndarray:
  """Applies random flips, brightness, contrast, hue changes, and rotation."""
  t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
  if torch.rand(1).item() > 0.5:
    t = TF.hflip(t)
  if torch.rand(1).item() > 0.5:
    t = TF.vflip(t)

  delta = (torch.rand(1).item() * 2 - 1) * 0.1
  t = (t + delta).clamp(0, 1)

  contrast_factor = 0.9 + torch.rand(1).item() * 0.1
  t = TF.adjust_contrast(t, contrast_factor)

  hue_delta = (torch.rand(1).item() * 2 - 1) * 0.05
  t = TF.adjust_hue(t, hue_delta)

  # +/- 10 degrees
  angle = (torch.rand(1).item() * 2 - 1) * 10.0
  t = TF.rotate(t, angle)

  return t.permute(1, 2, 0).numpy()
