from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from dlfb_pytorch.graphs.dataset.pairs import Pairs


@dataclass
class Graph:
  """Simple graph container with senders, receivers, and node IDs."""

  nodes: dict
  senders: torch.Tensor
  receivers: torch.Tensor
  n_node: int


@dataclass
class Dataset:
  """Graph dataset with nodes, pairs, and optional annotations."""

  n_nodes: int
  graph: Graph
  pairs: Pairs
  annotation: pd.DataFrame = field(default_factory=pd.DataFrame)

  def subset(
    self, node_ids: torch.Tensor, keep_original_ids: bool = True
  ) -> "Dataset":
    """Creates a subset of the dataset based on given node IDs."""
    if keep_original_ids:
      gid = node_ids
      n_nodes = self.n_nodes
    else:
      gid = torch.arange(node_ids.shape[0])
      n_nodes = node_ids.shape[0]

    lookup = {k: i for i, k in enumerate(node_ids.tolist())}
    graph = self.subset_graph(lookup, gid)
    pairs = self.subset_pairs(lookup)
    annotation = self.subset_annotation(lookup)
    return Dataset(n_nodes, graph, pairs, annotation)

  def subset_graph(self, lookup, gid) -> Graph:
    """Generates a subgraph by filtering nodes and reindexing edges."""
    ids = list(lookup.keys())
    senders_np = self.graph.senders.numpy()
    receivers_np = self.graph.receivers.numpy()
    edge_mask = np.isin(senders_np, ids) & np.isin(receivers_np, ids)
    graph = Graph(
      nodes={"gid": gid},
      senders=torch.tensor(
        [lookup[k] for k in senders_np[edge_mask].tolist()]
      ),
      receivers=torch.tensor(
        [lookup[k] for k in receivers_np[edge_mask].tolist()]
      ),
      n_node=len(ids),
    )
    return graph

  def subset_pairs(self, lookup: dict[int, int]) -> Pairs:
    """Subsets the positive and negative pairs by filtering and re-indexing."""
    ids = list(lookup.keys())
    pairs = {}

    for pair_type in ["pos", "neg"]:
      src = getattr(self.pairs, pair_type).numpy()
      pairs_mask = np.isin(src[:, 0], ids) & np.isin(src[:, 1], ids)
      pairs[pair_type] = torch.stack(
        [
          torch.tensor([lookup[k] for k in src[:, 0][pairs_mask].tolist()]),
          torch.tensor([lookup[k] for k in src[:, 1][pairs_mask].tolist()]),
        ],
        dim=1,
      )

    return Pairs(pos=pairs["pos"], neg=pairs["neg"])

  def subset_annotation(self, lookup: dict[int, int]) -> pd.DataFrame:
    """Subsets and reindexes node annotations using lookup dictionary."""
    lookup_df = pd.DataFrame(
      {"node_id": lookup.keys(), "new_node_id": lookup.values()}
    )
    annotation = pd.merge(lookup_df, self.annotation, on="node_id", how="left")
    annotation = annotation.drop(columns=["node_id"]).rename(
      columns={"new_node_id": "node_id"}
    )
    return annotation
