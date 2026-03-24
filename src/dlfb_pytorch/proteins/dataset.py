import os
from math import ceil

import numpy as np
import obonet
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from dlfb_pytorch.proteins.utils import get_device


def build_dataset(
  store_file_prefix: str, model_checkpoint: str, batch_size: int = 32
) -> dict[str, DataLoader]:
  """Build train/valid/test DataLoaders from stored embeddings."""
  dataset_splits = {}

  for split in ["train", "valid", "test"]:
    df = load_sequence_embeddings(
      store_file_prefix=f"{store_file_prefix}_{split}",
      model_checkpoint=model_checkpoint,
    )
    dataset_splits[split] = create_dataloader(
      df=df,
      is_training=(split == "train"),
      batch_size=batch_size,
    )
  return dataset_splits


def create_dataloader(
  df: pd.DataFrame,
  embeddings_prefix: str = "ME:",
  target_prefix: str = "GO:",
  is_training: bool = False,
  batch_size: int = 32,
) -> DataLoader:
  """Convert embedding DataFrame into a PyTorch DataLoader."""
  embeddings = torch.tensor(
    df.filter(regex=f"^{embeddings_prefix}").to_numpy(), dtype=torch.float32
  )
  targets = torch.tensor(
    df.filter(regex=f"^{target_prefix}").to_numpy(), dtype=torch.float32
  )
  dataset = TensorDataset(embeddings, targets)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=is_training,
    drop_last=is_training,
  )


def load_sequence_embeddings(
  store_file_prefix: str, model_checkpoint: str
) -> pd.DataFrame:
  """Load stored embedding DataFrame from disk."""
  model_name = model_checkpoint.replace("/", "_")
  store_file = f"{store_file_prefix}_{model_name}.feather"
  return pd.read_feather(store_file)


def store_sequence_embeddings(
  sequence_df: pd.DataFrame,
  store_prefix: str,
  tokenizer: PreTrainedTokenizer,
  model: PreTrainedModel,
  batch_size: int = 64,
  force: bool = False,
) -> None:
  """Extract and store mean embeddings for each protein sequence."""
  model_name = str(model.name_or_path).replace("/", "_")
  store_file = f"{store_prefix}_{model_name}.feather"

  if not os.path.exists(store_file) or force:
    device = get_device()

    n_batches = ceil(sequence_df.shape[0] / batch_size)
    batches: list[np.ndarray] = []
    for i in range(n_batches):
      batch_seqs = list(
        sequence_df["Sequence"][i * batch_size : (i + 1) * batch_size]
      )
      batches.extend(get_mean_embeddings(batch_seqs, tokenizer, model, device))

    embeddings = pd.DataFrame(np.vstack(batches))
    embeddings.columns = [f"ME:{int(i)+1}" for i in range(embeddings.shape[1])]
    df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
    df.to_feather(store_file)


def get_mean_embeddings(
  sequences: list[str],
  tokenizer: PreTrainedTokenizer,
  model: PreTrainedModel,
  device: torch.device | None = None,
) -> np.ndarray:
  """Compute mean embedding for each sequence using a protein LM."""
  if not device:
    device = get_device()

  model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
  model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

  model = model.to(device)
  model.eval()

  with torch.no_grad():
    outputs = model(**model_inputs)
    mean_embeddings = outputs.last_hidden_state.mean(dim=1)

  return mean_embeddings.detach().cpu().numpy()


def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
  """Return GO term to description mapping, downloading if needed."""
  if not os.path.exists(store_path):
    url = "https://current.geneontology.org/ontology/go-basic.obo"
    graph = obonet.read_obo(url)

    id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
    go_term_descriptions = pd.DataFrame(
      zip(id_to_name.keys(), id_to_name.values()),
      columns=["term", "description"],
    )
    go_term_descriptions.to_csv(store_path, index=False)

  else:
    go_term_descriptions = pd.read_csv(store_path)
  return go_term_descriptions
