import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(num_classes, **model_params):
  return LocalizationModel(num_classes=num_classes, **model_params)


def get_num_embeddings(model):
  return model.vector_quantizer.codebook.shape[1]


class LocalizationModel(nn.Module):
  """VQ-VAE model with a fully connected output head."""

  def __init__(
    self,
    embedding_dim: int,
    num_embeddings: int,
    commitment_cost: float,
    num_classes: int | None,
    dropout_rate: float,
    classification_head_layers: int,
  ):
    super().__init__()
    self.encoder = Encoder(latent_dim=embedding_dim)
    self.vector_quantizer = VectorQuantizer(
      num_embeddings=num_embeddings,
      embedding_dim=embedding_dim,
      commitment_cost=commitment_cost,
    )
    self.decoder = Decoder(latent_dim=embedding_dim)

    with torch.no_grad():
      dummy = torch.zeros(1, 1, 100, 100)
      ze = self.encoder(dummy)
      classification_input_dim = ze.shape[1] * ze.shape[2] * ze.shape[3]

    self.classification_head = ClassificationHead(
      input_dim=classification_input_dim,
      num_classes=num_classes,
      dropout_rate=dropout_rate,
      layers=classification_head_layers,
    )

  def forward(self, x, is_training: bool = True):
    """Runs a forward pass. Input: (B, C, H, W) in NCHW format."""
    ze = self.encoder(x)  # (B, D, H', W')
    ze_nhwc = ze.permute(0, 2, 3, 1)  # (B, H', W', D)
    zq_nhwc, perplexity, codebook_loss, commitment_loss = self.vector_quantizer(ze_nhwc)
    zq = zq_nhwc.permute(0, 3, 1, 2)  # (B, D, H', W')
    decoded = self.decoder(zq)  # (B, 1, H, W)
    logits = self.classification_head(zq_nhwc.reshape(zq_nhwc.shape[0], -1))
    return decoded, perplexity, codebook_loss, commitment_loss, logits

  def create_optimizer(self, lr: float = 0.001):
    """Creates an Adam optimizer for this model."""
    return torch.optim.Adam(self.parameters(), lr=lr)

  def get_encoding_indices(self, x):
    """Returns nearest codebook indices for input. Input: NCHW."""
    ze = self.encoder(x)
    ze_nhwc = ze.permute(0, 2, 3, 1)
    encoding_indices = self.vector_quantizer.get_closest_codebook_indices(ze_nhwc)
    return encoding_indices


class Encoder(nn.Module):
  """Convolutional encoder producing latent feature maps."""

  def __init__(self, latent_dim: int):
    super().__init__()
    self.conv1 = nn.Conv2d(1, latent_dim // 2, kernel_size=4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=4, stride=2, padding=1)
    self.conv3 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
    self.res_block1 = ResnetBlock(latent_dim)
    self.res_block2 = ResnetBlock(latent_dim)

  def forward(self, x):
    """Forward pass. Input/output: NCHW."""
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.conv3(x)
    x = self.res_block1(x)
    x = self.res_block2(x)
    return x


class ResnetBlock(nn.Module):
  """Residual convolutional block with GroupNorm and Swish activation."""

  def __init__(self, latent_dim: int):
    super().__init__()
    self.norm1 = nn.GroupNorm(32, latent_dim)
    self.conv1 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.GroupNorm(32, latent_dim)
    self.conv2 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    """Applies two conv layers with Swish activation and skip connection."""
    h = F.silu(self.norm1(x))
    h = self.conv1(h)
    h = F.silu(self.norm2(h))
    h = self.conv2(h)
    return x + h


class VectorQuantizer(nn.Module):
  """Vector quantization module for VQ-VAE."""

  def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.commitment_cost = commitment_cost
    self.codebook = nn.Parameter(torch.empty(embedding_dim, num_embeddings))
    nn.init.kaiming_uniform_(self.codebook)

  def forward(self, inputs):
    """Applies quantization. inputs: (B, H, W, D) in NHWC layout."""
    quantized, encoding_indices = self.quantize(inputs)
    codebook_loss, commitment_loss = self.compute_losses(inputs, quantized)
    perplexity = self.calculate_perplexity(encoding_indices)
    ste = self.get_straight_through_estimator(quantized, inputs)
    return ste, perplexity, codebook_loss, commitment_loss

  def quantize(self, inputs):
    """Snaps inputs to nearest codebook entries."""
    encoding_indices = self.get_closest_codebook_indices(inputs)
    flat_quantized = self.codebook[:, encoding_indices].t()
    quantized = flat_quantized.reshape(inputs.shape)
    return quantized, encoding_indices

  def get_closest_codebook_indices(self, inputs):
    """Returns indices of closest codebook vectors."""
    distances = self.calculate_distances(inputs)
    return torch.argmin(distances, dim=1)

  def calculate_distances(self, inputs):
    """Computes Euclidean distances between inputs and codebook vectors."""
    flat_inputs = inputs.reshape(-1, self.embedding_dim)
    distances = (
      torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
      - 2 * torch.matmul(flat_inputs, self.codebook)
      + torch.sum(self.codebook ** 2, dim=0, keepdim=True)
    )
    return distances

  def compute_losses(self, inputs, quantized):
    """Computes codebook and commitment losses."""
    codebook_loss = torch.mean((quantized - inputs.detach()) ** 2)
    commitment_loss = self.commitment_cost * torch.mean(
      (quantized.detach() - inputs) ** 2
    )
    return codebook_loss, commitment_loss

  def calculate_perplexity(self, encoding_indices):
    """Computes codebook usage perplexity."""
    encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
    avg_probs = encodings.mean(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity

  @staticmethod
  def get_straight_through_estimator(quantized, inputs):
    """Applies straight-through estimator to pass gradients through quantization."""
    return inputs + (quantized - inputs).detach()


class ClassificationHead(nn.Module):
  """Fully connected MLP head with optional dropout."""

  def __init__(self, input_dim: int, num_classes: int, dropout_rate: float, layers: int):
    super().__init__()
    modules = []
    in_features = input_dim
    for _ in range(layers - 1):
      modules.append(nn.Linear(in_features, 1000))
      modules.append(nn.ReLU())
      modules.append(nn.Dropout(dropout_rate))
      in_features = 1000
    modules.append(nn.Linear(in_features, num_classes))
    self.net = nn.Sequential(*modules)

  def forward(self, x):
    return self.net(x)


class Decoder(nn.Module):
  """Decoder module for reconstructing input from quantized representations."""

  def __init__(self, latent_dim: int):
    super().__init__()
    self.res_block1 = ResnetBlock(latent_dim)
    self.res_block2 = ResnetBlock(latent_dim)
    self.upsample1 = Upsample(latent_dim, latent_dim // 2, upfactor=2)
    self.upsample2 = Upsample(latent_dim // 2, 1, upfactor=2)

  def forward(self, x):
    """Applies the decoder. Input/output: NCHW."""
    x = self.res_block1(x)
    x = self.res_block2(x)
    x = self.upsample1(x)
    x = F.relu(x)
    x = self.upsample2(x)
    return x


class Upsample(nn.Module):
  """Upsampling block using bilinear interpolation followed by convolution."""

  def __init__(self, in_channels: int, out_channels: int, upfactor: int):
    super().__init__()
    self.upfactor = upfactor
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    """Upsamples input using bilinear interpolation and applies convolution."""
    x = F.interpolate(x, scale_factor=self.upfactor, mode="bilinear", align_corners=False)
    x = self.conv(x)
    return x
