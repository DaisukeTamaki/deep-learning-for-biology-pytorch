import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import LinkPropPredDataset

from dlfb_pytorch.graphs.dataset import Dataset, Graph
from dlfb_pytorch.graphs.dataset.pairs import Pairs


class DatasetBuilder:
  """Builds dataset splits for drug-drug interaction analysis and modeling."""

  def __init__(self, path):
    """Initializes the dataset builder with a path to the dataset."""
    self.path = path

  def build(
    self,
    node_limit: int | None = None,
    keep_original_ids: bool = False,
  ) -> dict[str, Dataset]:
    """Builds and returns a dictionary of dataset splits."""
    dataset_splits = {}
    n_nodes, split_pairs = self.download()
    annotation = self.prepare_annotation()

    for name, split in split_pairs.items():
      pos_pairs, neg_pairs = split["edge"], split["edge_neg"]
      graph = self.prepare_graph(n_nodes, pos_pairs)
      pairs = self.prepare_pairs(graph, pos_pairs, neg_pairs)
      dataset_splits.update({name: Dataset(n_nodes, graph, pairs, annotation)})

    if node_limit:
      dataset_splits = self.subset(
        dataset_splits, node_limit, keep_original_ids
      )

    return dataset_splits

  def download(self) -> tuple[int, dict]:
    """Downloads the dataset and returns the number of nodes and edge splits."""
    raw = LinkPropPredDataset(name="ogbl-ddi", root=self.path)
    n_nodes = raw[0]["num_nodes"]
    split_pairs = raw.get_edge_split()
    split_pairs["train"]["edge_neg"] = None
    return n_nodes, split_pairs

  def prepare_annotation(self) -> pd.DataFrame:
    """Annotates nodes by mapping node IDs to database IDs and drug names."""
    ddi_descriptions = pd.read_csv(
      f"{self.path}/ogbl_ddi/mapping/ddi_description.csv.gz"
    )
    node_to_dbid_lookup = pd.read_csv(
      f"{self.path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    )
    first_drug = ddi_descriptions.loc[
      :, ["first drug id", "first drug name"]
    ].rename(columns={"first drug id": "dbid", "first drug name": "drug_name"})

    second_drug = ddi_descriptions.loc[
      :, ["second drug id", "second drug name"]
    ].rename(
      columns={"second drug id": "dbid", "second drug name": "drug_name"}
    )
    dbid_to_name_lookup = (
      pd.concat([first_drug, second_drug])
      .drop_duplicates()
      .reset_index(drop=True)
    )

    annotation = pd.merge(
      node_to_dbid_lookup.rename(
        columns={"drug id": "dbid", "node idx": "node_id"}
      ),
      dbid_to_name_lookup,
      on="dbid",
      how="inner",
    )
    return annotation

  def prepare_graph(self, n_nodes: int, pos_pairs) -> Graph:
    """Prepares a graph from positive edge pairs."""
    pos_pairs_t = torch.tensor(pos_pairs) if not isinstance(pos_pairs, torch.Tensor) else pos_pairs
    senders, receivers = self.make_undirected(pos_pairs_t[:, 0], pos_pairs_t[:, 1])
    graph = Graph(
      nodes={"gid": torch.arange(n_nodes)},
      senders=senders,
      receivers=receivers,
      n_node=n_nodes,
    )
    return graph

  @staticmethod
  def make_undirected(
    senders: torch.Tensor, receivers: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Makes an undirected graph by duplicating edges in both directions."""
    senders_undir = torch.cat((senders, receivers))
    receivers_undir = torch.cat((receivers, senders))
    return senders_undir, receivers_undir

  def prepare_pairs(
    self, graph: Graph, pos_pairs, neg_pairs=None
  ) -> Pairs:
    """Prepares positive and negative edge pairs."""
    pos_t = torch.tensor(pos_pairs) if not isinstance(pos_pairs, torch.Tensor) else pos_pairs
    if neg_pairs is None:
      neg_t = self.infer_negative_pairs(graph)
    else:
      neg_t = torch.tensor(neg_pairs) if not isinstance(neg_pairs, torch.Tensor) else neg_pairs
    return Pairs(pos=pos_t, neg=neg_t)

  def infer_negative_pairs(self, graph: Graph) -> torch.Tensor:
    """Infers negative edge pairs in a graph."""
    neg_adj_mask = np.ones((graph.n_node, graph.n_node), dtype=np.uint8)
    neg_adj_mask[graph.senders.numpy(), graph.receivers.numpy()] = 0
    neg_adj_mask = np.triu(neg_adj_mask, k=1)
    neg_pairs = torch.tensor(np.array(neg_adj_mask.nonzero()).T)
    return neg_pairs

  def subset(
    self,
    dataset_splits: dict[str, Dataset],
    node_limit: int,
    keep_original_ids: bool = False,
  ) -> dict[str, Dataset]:
    """Creates subset of dataset splits by sampling a fixed number of nodes."""
    perm = torch.randperm(dataset_splits["train"].n_nodes)[:node_limit]
    node_ids = perm.sort().values

    dataset_subset_splits = {}
    for name, dataset in dataset_splits.items():
      dataset_subset_splits[name] = dataset.subset(node_ids, keep_original_ids)

    return dataset_subset_splits
