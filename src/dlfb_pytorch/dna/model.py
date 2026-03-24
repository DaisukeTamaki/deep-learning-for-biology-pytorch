import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
  """Basic CNN model for binary sequence classification."""

  def __init__(self, input_channels: int = 4, conv_filters: int = 64,
               kernel_size: int = 10, dense_units: int = 128,
               seq_len: int = 200):
    super().__init__()
    self.conv1 = nn.Conv1d(input_channels, conv_filters, kernel_size, padding="same")
    self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size, padding="same")
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    flat_dim = conv_filters * (seq_len // 4)
    self.fc1 = nn.Linear(flat_dim, dense_units)
    self.fc2 = nn.Linear(dense_units, dense_units // 2)
    self.out = nn.Linear(dense_units // 2, 1)

  def forward(self, x):
    # Input x: (batch, seq_len, channels) -> transpose to (batch, channels, seq_len)
    x = x.permute(0, 2, 1)

    x = self.pool(F.gelu(self.conv1(x)))
    x = self.pool(F.gelu(self.conv2(x)))

    x = x.reshape(x.shape[0], -1)
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))
    return self.out(x)


class ConvModelV2(nn.Module):
  """CNN with batch norm and dropout for binary classification."""

  def __init__(self, input_channels: int = 4, conv_filters: int = 64,
               kernel_size: int = 10, dense_units: int = 128,
               dropout_rate: float = 0.2, seq_len: int = 200):
    super().__init__()
    self.conv1 = nn.Conv1d(input_channels, conv_filters, kernel_size, padding="same")
    self.bn1 = nn.BatchNorm1d(conv_filters)
    self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size, padding="same")
    self.bn2 = nn.BatchNorm1d(conv_filters)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    flat_dim = conv_filters * (seq_len // 4)
    self.fc1 = nn.Linear(flat_dim, dense_units)
    self.drop1 = nn.Dropout(dropout_rate)
    self.fc2 = nn.Linear(dense_units, dense_units // 2)
    self.drop2 = nn.Dropout(dropout_rate)
    self.out = nn.Linear(dense_units // 2, 1)

  def forward(self, x):
    x = x.permute(0, 2, 1)

    x = self.pool(F.gelu(self.bn1(self.conv1(x))))
    x = self.pool(self.bn2(F.gelu(self.conv2(x))))

    x = x.reshape(x.shape[0], -1)
    x = self.drop1(F.gelu(self.fc1(x)))
    x = self.drop2(F.gelu(self.fc2(x)))
    return self.out(x)


class ConvBlock(nn.Module):
  """Convolutional block with batch norm, GELU and max pooling."""

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: int = 10, pool_size: int = 2):
    super().__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
    self.bn = nn.BatchNorm1d(out_channels)
    self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

  def forward(self, x):
    x = self.pool(F.gelu(self.bn(self.conv(x))))
    return x


class TransformerBlock(nn.Module):
  """Transformer block with self-attention and MLP."""

  def __init__(self, embed_dim: int, num_heads: int = 8,
               dense_units: int = 64, dropout_rate: float = 0.2):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.fc1 = nn.Linear(embed_dim, dense_units)
    self.drop = nn.Dropout(dropout_rate)
    self.fc2 = nn.Linear(dense_units, embed_dim)

  def forward(self, x):
    # x: (batch, seq_len, embed_dim)
    residual = x
    x = self.norm1(x)
    x, _ = self.attn(x, x, x)
    x = x + residual

    residual = x
    x = self.norm2(x)
    x = self.drop(F.gelu(self.fc1(x)))
    x = self.fc2(x)
    x = x + residual
    return x


class MLPBlock(nn.Module):
  """Dense + GELU + dropout block."""

  def __init__(self, in_features: int, out_features: int,
               dropout_rate: float = 0.0):
    super().__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    return self.drop(F.gelu(self.fc(x)))


class ConvTransformerModel(nn.Module):
  """Model combining CNN, transformer, and MLP blocks."""

  def __init__(self, input_channels: int = 4, num_conv_blocks: int = 2,
               conv_filters: int = 64, kernel_size: int = 10,
               num_mlp_blocks: int = 2, dense_units: int = 128,
               dropout_rate: float = 0.2, num_transformer_blocks: int = 0,
               num_transformer_heads: int = 8,
               transformer_dense_units: int = 64,
               seq_len: int = 200):
    super().__init__()

    self.conv_blocks = nn.ModuleList()
    in_ch = input_channels
    for _ in range(num_conv_blocks):
      self.conv_blocks.append(
        ConvBlock(in_ch, conv_filters, kernel_size, pool_size=2)
      )
      in_ch = conv_filters

    self.transformer_blocks = nn.ModuleList()
    for _ in range(num_transformer_blocks):
      self.transformer_blocks.append(
        TransformerBlock(conv_filters, num_transformer_heads,
                         transformer_dense_units, dropout_rate)
      )

    flat_dim = conv_filters * (seq_len // (2 ** num_conv_blocks))

    self.mlp_blocks = nn.ModuleList()
    in_dim = flat_dim
    for i in range(num_mlp_blocks):
      out_dim = dense_units // (i + 1)
      self.mlp_blocks.append(MLPBlock(in_dim, out_dim, dropout_rate))
      in_dim = out_dim

    self.out = nn.Linear(in_dim, 1)

  def forward(self, x):
    # (batch, seq_len, channels) -> (batch, channels, seq_len)
    x = x.permute(0, 2, 1)

    for conv_block in self.conv_blocks:
      x = conv_block(x)

    if self.transformer_blocks:
      # (batch, channels, seq_len) -> (batch, seq_len, channels)
      x = x.permute(0, 2, 1)
      for transformer_block in self.transformer_blocks:
        x = transformer_block(x)
      # Back to (batch, channels, seq_len) for flattening
      x = x.permute(0, 2, 1)

    x = x.reshape(x.shape[0], -1)

    for mlp_block in self.mlp_blocks:
      x = mlp_block(x)

    return self.out(x)
