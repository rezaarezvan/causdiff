import math
import torch

from torch import nn


class Block(nn.Module):
    """
    A neural network block with linear layers and SiLU activation.

    Args:
        inter_dim (int): Dimension of the block.
    """

    def __init__(self, inter_dim=512):
        super().__init__()
        self.ff = nn.Linear(inter_dim, inter_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.act(self.ff(x))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with linear layers and SiLU (Swish) activation.
    """

    def __init__(self, dim=2, layers=5, inter_dim=512, dim_t=512):
        """
        Initializes the MLP.

        Args:
            dim (int): Input and output dimensionality.
            layers (int): Number of layers.
            inter_dim (int): Hidden layer dimensionality.
            dim_t (int): Dimensionality of the time embedding.
        """
        super().__init__()
        self.channels_t = dim_t
        self.in_projection = nn.Linear(dim, inter_dim)
        self.t_projection = nn.Linear(dim_t, inter_dim)
        self.blocks = nn.Sequential(*[Block(inter_dim) for _ in range(layers)])
        self.out_projection = nn.Linear(inter_dim, dim)

    def gen_t_embedding(self, t, max_positions=10_000):
        """
        Generates a (sine, cosine) time embedding. The time embedding is a positional
        encoding that is added to the input tensor. The motivation behind this is to
        allow the model to learn a time-dependent function.

        The time embedding is generated using the formula,

            PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}}) - even indices
            PE_{(pos, 2i + 1)} = cos(pos / 10000^{2i / d_{model}}) - odd indices

        where PE is the positional encoding, pos is the position i is the
        dimension index, and d_{model} is the model dimensionality.

        Args:
            t (torch.Tensor): Time tensor.
            max_positions (int): Maximum number of positions.

        Returns:
            torch.Tensor: Time embedding.
        """

        # Generate all the possible positions
        t = t * max_positions

        # Split dimensionality in half
        half_dim = self.channels_t // 2

        # Generate the time embedding using the formula above
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        # If the dimensionality is odd, zero pad the time embedding
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def forward(self, x, t, _label):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.
            _label (torch.Tensor): Label tensor (not used, API compatibility).

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x
