import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


def skew(image: np.ndarray, size: tuple[int, int, int] = (224, 224, 3)) -> np.ndarray:
  """Rescales and resizes image to fixed size using bilinear interpolation."""
  image = rescale_image(image)
  t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
  t = F.interpolate(t, size=(size[0], size[1]), mode="bilinear", align_corners=False)
  return t.squeeze(0).permute(1, 2, 0).numpy()


def crop(image: np.ndarray) -> np.ndarray:
  """Rescales, resizes with preserved aspect ratio, then center-crops image."""
  image = rescale_image(image)
  image = resize_preserve_aspect(image, 256)
  image = center_crop(image, 224)
  return image


def resize_preserve_aspect(image: np.ndarray, short_side: int = 256) -> np.ndarray:
  """Resize image with shorter side is `short_side`, keeping aspect ratio."""
  h, w, c = image.shape
  scale = short_side / min(h, w)
  new_h = round(h * scale)
  new_w = round(w * scale)
  t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
  t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
  return t.squeeze(0).permute(1, 2, 0).numpy()


def center_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
  """Crop the center square of given size from an image."""
  h, w, _ = image.shape
  top = (h - size) // 2
  left = (w - size) // 2
  return image[top : top + size, left : left + size]


def rescale_image(image: np.ndarray) -> np.ndarray:
  """Normalizes pixel values to the [0, 1] range by dividing by 255."""
  return image / 255.0


def resnet(image: np.ndarray) -> np.ndarray:
  """Preprocess using the pretrained ResNet image processor."""
  processed = IMAGE_PROCESSOR(image, return_tensors="np", do_rescale=True)
  pixel_values = processed["pixel_values"]  # (1, C, H, W)
  return np.transpose(pixel_values[0], (1, 2, 0))  # CHW -> HWC for memmap


def skew_resnet(image: np.ndarray) -> np.ndarray:
  """Applies skew resizing followed by ResNet normalization."""
  image = skew(image)
  image = resnet_normalize_image(image)
  return image


def crop_resnet(image: np.ndarray) -> np.ndarray:
  """Applies center crop followed by ResNet normalization."""
  image = crop(image)
  image = resnet_normalize_image(image)
  return image


def resnet_normalize_image(image: np.ndarray) -> np.ndarray:
  """Applies ResNet-style normalization using fixed mean and std."""
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  return (image - mean) / std
