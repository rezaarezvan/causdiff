"""
UNet model specifically designed for Denoising Diffusion Probabilistic Models (DDPM).

This module provides a UNet model that takes noisy images and timesteps as input
and predicts the noise component. The architecture follows the design principles
from Ho et al. (2020) and Nichol & Dhariwal (2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps in diffusion models.

    Args:
        dim (int): Dimension of the embedding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Forward pass to compute the sinusoidal embeddings for timesteps.

        Args:
            time (torch.Tensor): Timesteps to embed, shape [batch_size]

        Returns:
            Timestep embeddings, shape [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Zero-pad if dimension is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0), mode="constant")

        return embeddings


class Block(nn.Module):
    """
    Basic convolutional block with residual connection.

    Args:
        dim (int): Input channel dimension
        dim_out (int): Output channel dimension
        groups (int): Number of groups for GroupNorm
        dropout (float): Dropout probability
    """

    def __init__(self, dim, dim_out, groups=8, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor
            scale_shift (tuple, optional): Scale and shift parameters for normalization

        Returns:
            Output tensor after processing
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(nn.Module):
    """
    Residual block with time embeddings.

    Args:
        dim (int): Input dimension
        dim_out (int): Output dimension
        time_emb_dim (int): Time embedding dimension
        groups (int): Number of groups for GroupNorm
        dropout (float): Dropout probability
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Forward pass through the ResNet block.

        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor, optional): Time embedding tensor

        Returns:
            Output tensor after processing
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    Self-attention module for diffusion models.

    Args:
        dim (int): Input dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension of each head
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Forward pass through the attention module.

        Args:
            x (torch.Tensor): Input tensor [b, c, h, w]

        Returns:
            Output tensor after attention
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h * w).transpose(2, 3), qkv
        )

        q = q * self.scale
        sim = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(sim, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(2, 3).reshape(b, -1, h, w)

        return self.to_out(out)


class Upsample(nn.Module):
    """
    Upsampling module.

    Args:
        dim (int): Input dimension
        dim_out (int, optional): Output dimension, defaults to None
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass with upsampling and convolution.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            Upsampled tensor
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """
    Downsampling module.

    Args:
        dim (int): Input dimension
        dim_out (int, optional): Output dimension, defaults to None
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Forward pass with downsampling via strided convolution.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            Downsampled tensor
        """
        return self.conv(x)


class DDPMUNet(nn.Module):
    """
    UNet model for DDPM, designed to predict noise from noisy images and timesteps.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        hidden_dims (list): Dimensions for each UNet level
        dim_mults (list): Dimension multipliers for each level
        attention_resolutions (list): Resolutions to apply attention
        dropout (float): Dropout probability
        num_res_blocks (int): Number of residual blocks per level
        time_emb_dim_mult (int): Multiplier for time embedding dimension
        use_attention (bool): Whether to use attention blocks
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_dims=64,
        dim_mults=(1, 2, 4, 8),
        attention_resolutions=(8, 16, 32),
        dropout=0.1,
        num_res_blocks=2,
        time_emb_dim_mult=4,
        use_attention=True,
    ):
        super().__init__()

        # Time embeddings
        time_dim = hidden_dims * time_emb_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims),
            nn.Linear(hidden_dims, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Define dimensions for each level
        dims = [hidden_dims]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Create dimensions list from multipliers
        dims = [hidden_dims * m for m in dim_mults]
        dims = [hidden_dims, *dims]

        in_out = list(zip(dims[:-1], dims[1:]))

        # Input projection
        self.init_conv = nn.Conv2d(in_channels, hidden_dims, kernel_size=3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            resolution = hidden_dims // (2**i)
            use_attn = resolution in attention_resolutions and use_attention

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in, dim_in, time_emb_dim=time_dim, dropout=dropout
                        ),
                        ResnetBlock(
                            dim_in, dim_in, time_emb_dim=time_dim, dropout=dropout
                        ),
                        Attention(dim_in) if use_attn else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout
        )
        self.mid_attn = Attention(mid_dim) if use_attention else nn.Identity()
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, dropout=dropout
        )

        # Upsampling blocks
        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            resolution = hidden_dims // (2 ** (len(in_out) - i - 1))
            use_attn = resolution in attention_resolutions and use_attention

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            dropout=dropout,
                        ),
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            dropout=dropout,
                        ),
                        Attention(dim_out) if use_attn else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # Output projection
        self.final_res_block = ResnetBlock(
            hidden_dims * 2, hidden_dims, time_emb_dim=time_dim, dropout=dropout
        )
        self.final_conv = nn.Conv2d(hidden_dims, out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Forward pass to predict noise.

        Args:
            x (torch.Tensor): Noisy input image [B, C, H, W]
            t (torch.Tensor): Timesteps [B]

        Returns:
            Predicted noise with same shape as input
        """
        # Get time embeddings
        t_emb = self.time_mlp(t)

        # Initial projection
        h = self.init_conv(x)
        h_stack = [h]

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            h = attn(h)
            h_stack.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling with skip connections
        for block1, block2, attn, upsample in self.ups:
            h = torch.cat([h, h_stack.pop()], dim=1)
            h = block1(h, t_emb)
            h = torch.cat([h, h_stack.pop()], dim=1)
            h = block2(h, t_emb)
            h = attn(h)
            h = upsample(h)

        # Push through final blocks
        h = torch.cat([h, h_stack.pop()], dim=1)
        h = self.final_res_block(h, t_emb)

        return self.final_conv(h)
