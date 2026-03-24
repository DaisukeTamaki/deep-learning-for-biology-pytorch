import numpy as np
import torch
from matplotlib import pyplot as plt

from dlfb_pytorch.localization.dataset import AnnotatedFrame, Dataset
from dlfb_pytorch.localization.model import LocalizationModel
from dlfb_pytorch.localization.utils import calculate_grid_dimensions


def show_reconstruction(dataset: Dataset, model: LocalizationModel, n: int):
  gene_symbols, localizations, images = [], [], []
  annotated_frame: AnnotatedFrame
  for annotated_frame in dataset.get_random_annotated_frames(n=n):
    gene_symbols.append(annotated_frame.gene_symbol)
    localizations.append(annotated_frame.localization)
    images.append(annotated_frame.image)

  images_np = np.array(images)
  images_tensor = (
    torch.from_numpy(np.ascontiguousarray(images_np))
    .permute(0, 3, 1, 2)
    .float()
  )
  device = next(model.parameters()).device
  model.eval()
  with torch.no_grad():
    reconstructed, _, _, _, _ = model(images_tensor.to(device), is_training=False)

  reconstructed_images = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
  image_pairs_with_labels = [
    ((i, r), f"{g}: {lo}")
    for i, r, g, lo in zip(
      images, reconstructed_images, gene_symbols, localizations
    )
  ]
  fig = plot_combined_images(image_pairs_with_labels)
  return fig


def plot_combined_images(image_pairs_with_labels):
  num_pairs = len(image_pairs_with_labels)
  nrows, ncols = calculate_grid_dimensions(num_pairs, ratio=0.5)

  fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 2)
  )

  if nrows == 1 or ncols == 1:
    axs = np.array(axs).reshape(nrows, ncols)

  for i, (images, label) in enumerate(image_pairs_with_labels):
    row, col = divmod(i, ncols)
    combined_image = np.concatenate(images, axis=1)
    height, width = combined_image.shape[:2]
    axs[row, col].imshow(combined_image, cmap="binary_r")
    axs[row, col].axis("off")
    axs[row, col].set_title(
      label,
      fontsize=10,
      x=0.5,
      y=0.9,
      color="black",
      ha="center",
      va="top",
      bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3"),
    )
    axs[row, col].text(
      width * 0.05,
      height * 0.9,
      "O",
      color="black",
      ha="center",
      fontsize=8,
    )
    axs[row, col].text(
      width * 0.95,
      height * 0.9,
      "R",
      color="black",
      ha="center",
      fontsize=8,
    )

  for ax in axs.flat[i + 1 :]:
    ax.axis("off")

  plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
  return fig
