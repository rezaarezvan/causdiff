import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear_1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear_2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, t):
        """
        Expects t of shape (batch_size,)
        """
        assert t.dim() == 1, "Time tensor must be 1D"
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        embeddings = self.linear_1(embeddings)
        embeddings = F.silu(embeddings)
        embeddings = self.linear_2(embeddings)

        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        scale = 1 / math.sqrt(head_dim)
        attn = torch.einsum("bchp,bchq->bpq", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bpq,bchq->bchp", attn, v)
        out = out.view(b, c, h, w)

        return self.proj_out(out) + x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb=None):
        identity = self.shortcut(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        if time_emb is not None and self.time_mlp is not None:
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            x = x + time_emb

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + identity


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, add_attention=False):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.attention = AttentionBlock(out_channels) if add_attention else None

    def forward(self, x, time_emb):
        x = self.conv(x, time_emb)
        if self.attention is not None:
            x = self.attention(x)
        return self.downsample(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, add_attention=False):
        super().__init__()
        self.conv = ConvBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.attention = AttentionBlock(out_channels) if add_attention else None

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x, time_emb)

        if self.attention is not None:
            x = self.attention(x)

        return x


class ConditionalClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, class_labels):
        return self.embedding(class_labels)


class UNetWithAttention(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        out_channels=3,
        model_channels=32,
        num_classes=102,
        channel_multipliers=(1, 1, 2, 2, 4),
        attention_resolutions=(8, 16, 32),
        time_emb_dim=256,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.class_embedding = ConditionalClassEmbedding(num_classes, time_emb_dim)

        # Input convolution
        self.input_conv = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1
        )

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]

        input_ch = model_channels
        resolution = img_size

        for i, mult in enumerate(channel_multipliers):
            output_ch = model_channels * mult
            add_attn = resolution in attention_resolutions

            self.down_blocks.append(
                DownBlock(input_ch, output_ch, time_emb_dim, add_attention=add_attn)
            )

            input_ch = output_ch
            resolution = resolution // 2
            channels.append(output_ch)

        # Middle block
        self.middle_block = nn.Sequential(
            ConvBlock(channels[-1], channels[-1], time_emb_dim),
            AttentionBlock(channels[-1]),
            ConvBlock(channels[-1], channels[-1], time_emb_dim),
        )

        self.up_blocks = nn.ModuleList()

        # We need to process the up blocks in reverse order
        for i in range(len(channel_multipliers)):
            # This is the output channel dimension from the corresponding down block
            # -1 is bottleneck, -2 is first skip, etc.
            skip_channels = channels[-(i + 2)]

            # The input channels to the up block come from the previous layer's output
            in_channels = channels[-1] if i == 0 else channels[-(i + 1)]

            resolution = resolution * 2
            add_attn = resolution in attention_resolutions

            self.up_blocks.append(
                UpBlock(
                    in_channels, skip_channels, time_emb_dim, add_attention=add_attn
                )
            )

        # Final convolution - input is the output of last up block which should match first down block channels
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, label=None, source_label=None, target_label=None):
        # Time embedding
        t_emb = self.time_embedding(t)

        # Class conditioning
        if source_label is not None and target_label is not None:
            source_emb = self.class_embedding(source_label)
            target_emb = self.class_embedding(target_label)
            t_emb = t_emb + source_emb + target_emb
        elif label is not None:
            t_emb = t_emb + self.class_embedding(label)

        # Input convolution
        h = self.input_conv(x)

        # Store skip connections
        skips = [h]

        # Downsampling
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        # Middle block
        h = self.middle_block[0](h, t_emb)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, t_emb)

        # Upsampling
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            h = block(h, skip, t_emb)

        # Final output
        return self.output_conv(h)
