import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim=256, max_positions=10000):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # An MLP that maps the sinusoidal embedding to time_emb_dim
        self.lin1 = nn.Linear(time_emb_dim, time_emb_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(time_emb_dim, time_emb_dim)
        self.max_positions = max_positions

    def sinusoidal_embedding(self, t):
        """t is in [0, 1]; scale it by max_positions for typical diffusion-style embedding."""
        t_scaled = t * self.max_positions
        half_dim = self.time_emb_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        # if time_emb_dim is odd, pad
        if self.time_emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0)
        return emb

    def forward(self, t):
        """
        Expects t of shape (batch_size,)
        Returns an embedding of shape (batch_size, time_emb_dim)
        """
        emb = self.sinusoidal_embedding(t)
        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)
        return emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class TimeAwareDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Linear layer that maps the time embedding to (out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.act = nn.SiLU()  # or ReLU

    def forward(self, x, t_emb):
        """
        x: shape (B, in_channels, H, W)
        t_emb: shape (B, time_emb_dim)
        """
        # 1) Project time embedding to shape (B, out_channels)
        t_out = self.time_emb_proj(t_emb)
        # 2) Reshape to (B, out_channels, 1, 1) so we can broadcast-add
        t_out = t_out[:, :, None, None]

        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        # inject time embedding as a bias
        x = x + t_out
        x = self.act(x)

        # conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = TimeAwareDoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.pool(x)
        x = self.conv(x, t_emb)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = TimeAwareDoubleConv(in_channels, out_channels, time_emb_dim)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = TimeAwareDoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, t_emb)
        return x


class FlowUNet(nn.Module):
    def __init__(self, channels=1, n_classes=1, time_emb_dim=256, bilinear=True):
        """
        channels: input channels (e.g., 1 for MNIST)
        n_classes:  number of classes for label embedding
        time_emb_dim: dimension of the time embedding
        bilinear: whether to use bilinear interpolation in the upsampling layers
        """
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, time_emb_dim)
        self.source_label_emb = nn.Embedding(n_classes, time_emb_dim)
        self.target_label_emb = nn.Embedding(n_classes, time_emb_dim)

        self.time_mlp = TimeEmbedding(time_emb_dim=time_emb_dim)

        self.inc = TimeAwareDoubleConv(channels, 64, time_emb_dim)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, time_emb_dim)

        self.up1 = Up(1024, 512 // factor, time_emb_dim, bilinear)
        self.up2 = Up(512, 256 // factor, time_emb_dim, bilinear)
        self.up3 = Up(256, 128 // factor, time_emb_dim, bilinear)
        self.up4 = Up(128, 64, time_emb_dim, bilinear)
        self.outc = nn.Conv2d(64, channels, kernel_size=1)

    def forward(self, x, t, label=None, source_label=None, target_label=None):
        """
        x: shape (B, n_channels, H, W)
        t: shape (B,) - e.g. random in [0,1]
        Conditioning:
            - For one-sided conditioning, pass a tensor 'label'
            - For two-sided conditioning, pass both 'source_label' and 'target_label'
        """
        # 1) get the time embedding
        t_emb = self.time_mlp(t)  # shape (B, time_emb_dim)
        if self.label_emb.num_embeddings > 1:
            if source_label is not None and target_label is not None:
                #  Two-sided conditioning: add both source and target label embeddings
                t_emb = (
                    t_emb
                    + self.source_label_emb(source_label)
                    + self.target_label_emb(target_label)
                )
            elif label is not None:
                # One-sided conditioning: add single label embedding
                t_emb = t_emb + self.label_emb(label)
            # else: no conditioning added

        # 2) pass through the UNet
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        logits = self.outc(x)
        return logits
