import torch
import torch.nn as nn
import torch.nn.functional as F


class DdiModel(nn.Module):
  """Graph-based model for predicting drug-drug interactions (DDIs)."""

  def __init__(
    self,
    n_nodes: int,
    embedding_dim: int,
    dropout_rate: float,
    last_layer_self: bool,
    degree_norm: bool,
    n_mlp_layers: int = 2,
  ):
    super().__init__()
    self.node_encoder = NodeEncoder(
      n_nodes,
      embedding_dim,
      last_layer_self,
      degree_norm,
      dropout_rate,
    )
    self.link_predictor = LinkPredictor(
      embedding_dim, n_mlp_layers, dropout_rate
    )

  def forward(self, graph, pairs, is_pred: bool = False):
    """Generates interaction scores for node pairs."""
    h = self.node_encoder(graph)

    if is_pred:
      scores = self.link_predictor(h[pairs[:, 0]], h[pairs[:, 1]])
    else:
      pos_senders, pos_receivers = pairs["pos"][:, 0], pairs["pos"][:, 1]
      neg_senders, neg_receivers = pairs["neg"][:, 0], pairs["neg"][:, 1]
      scores = {
        "pos": self.link_predictor(h[pos_senders], h[pos_receivers]),
        "neg": self.link_predictor(h[neg_senders], h[neg_receivers]),
      }
    return scores

  def create_optimizer(self, lr: float = 0.001):
    """Creates an Adam optimizer for this model."""
    return torch.optim.Adam(self.parameters(), lr=lr)

  @staticmethod
  def add_mean_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    """Concatenates a mean embedding to the existing embeddings."""
    mean_embeddings = embeddings.mean(dim=0, keepdim=True)
    return torch.cat([embeddings, mean_embeddings], dim=0)


class NodeEncoder(nn.Module):
  """Encodes nodes into embeddings using a two-layer GraphSAGE model."""

  def __init__(
    self,
    n_nodes: int,
    embedding_dim: int,
    last_layer_self: bool,
    degree_norm: bool,
    dropout_rate: float,
  ):
    super().__init__()
    self.last_layer_self = last_layer_self
    self.degree_norm = degree_norm
    self.dropout_rate = dropout_rate

    self.node_embeddings = nn.Embedding(
      num_embeddings=n_nodes,
      embedding_dim=embedding_dim,
    )
    nn.init.xavier_uniform_(self.node_embeddings.weight)

    self.conv1 = SAGEConv(embedding_dim, embedding_dim, with_self=True, degree_norm=degree_norm)
    self.conv2 = SAGEConv(embedding_dim, embedding_dim, with_self=last_layer_self, degree_norm=degree_norm)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, graph) -> torch.Tensor:
    """Encodes the nodes of a graph into embeddings."""
    x = self.node_embeddings(graph.nodes["gid"])
    x = self.conv1(graph, x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv2(graph, x)
    return x


class SAGEConv(nn.Module):
  """GraphSAGE convolutional layer with optional self-loops."""

  def __init__(self, in_dim: int, out_dim: int, with_self: bool, degree_norm: bool):
    super().__init__()
    self.with_self = with_self
    self.degree_norm = degree_norm
    self.linear = nn.Linear(in_dim * 2, out_dim)

  def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
    n_nodes = x.shape[0]

    if self.with_self:
      senders, receivers = self._add_self_edges(graph, n_nodes)
    else:
      senders, receivers = graph.senders, graph.receivers

    if not self.degree_norm:
      x_updated = self._segment_mean(x[senders], receivers, n_nodes)
    else:
      sender_degree = self._get_degree(senders, n_nodes)
      x_norm = self._normalize_by_degree(x, sender_degree)
      x_updated = self._segment_mean(x_norm[senders], receivers, n_nodes)
      receiver_degree = self._get_degree(receivers, n_nodes)
      x_updated = self._normalize_by_degree(x_updated, receiver_degree)

    combined_embeddings = torch.cat([x, x_updated], dim=-1)
    return self.linear(combined_embeddings)

  @staticmethod
  def _add_self_edges(graph, n_nodes: int):
    """Adds self-loops to the graph."""
    all_nodes = torch.arange(n_nodes, device=graph.senders.device)
    senders = torch.cat([graph.senders, all_nodes])
    receivers = torch.cat([graph.receivers, all_nodes])
    return senders, receivers

  @staticmethod
  def _segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Computes mean of data for each segment (scatter mean)."""
    out = torch.zeros(num_segments, data.shape[-1], dtype=data.dtype, device=data.device)
    count = torch.zeros(num_segments, 1, dtype=data.dtype, device=data.device)
    out.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
    ones = torch.ones(data.shape[0], 1, dtype=data.dtype, device=data.device)
    count.scatter_add_(0, segment_ids.unsqueeze(-1), ones)
    return out / count.clamp(min=1)

  @staticmethod
  def _get_degree(indices: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Computes node degrees from edge indices."""
    return torch.zeros(n_nodes, device=indices.device).scatter_add_(
      0, indices, torch.ones_like(indices, dtype=torch.float)
    )

  @staticmethod
  def _normalize_by_degree(x: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    """Normalizes node features by the square root of the degree."""
    return x * torch.rsqrt(degree.clamp(min=1.0)).unsqueeze(-1)


class LinkPredictor(nn.Module):
  """Predicts interaction scores for pairs of node embeddings."""

  def __init__(self, embedding_dim: int, n_layers: int, dropout_rate: float):
    super().__init__()
    layers = []
    for i in range(n_layers - 1):
      layers.append(nn.Linear(embedding_dim, embedding_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(embedding_dim, 1))
    self.net = nn.Sequential(*layers)

  def forward(
    self,
    sender_embeddings: torch.Tensor,
    receiver_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    """Computes scores for node pairs."""
    x = sender_embeddings * receiver_embeddings
    return self.net(x).squeeze(-1)
