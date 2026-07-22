"""ExU-Trans approximation used by this repository.

This is a lightweight 2D implementation inspired by the paper's dual-encoder
U-Net + Transformer design. It is not claimed to be a bit-for-bit copy of the
published model because the paper does not specify all architectural widths,
patch settings, split IDs, or training details needed for exact reproduction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)


class SEMHABlock(nn.Module):
    """Approximation of Self-Explanatory Multi-Head Attention (SE-MHA)."""

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.explanation_residual = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.n2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.attn_map = None

    def forward(self, x):
        x1 = self.n1(x)
        attn_out, attn_w = self.attn(x1, x1, x1, need_weights=True, average_attn_weights=False)
        x = x + attn_out + 0.1 * self.explanation_residual(x1)
        x = x + self.mlp(self.n2(x))
        self.attn_map = attn_w.mean(dim=1)
        return x


class DAE(nn.Module):
    """Approximation of the paper's Discriminative Attribute Explainer (DAE)."""

    def __init__(self, embed_dim: int, attr_dim: int = 4) -> None:
        super().__init__()
        self.token_score = nn.Linear(embed_dim, 1)
        self.attr_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, attr_dim))

    def forward(self, tokens, hw):
        attr_logits = self.attr_head(tokens.mean(dim=1))
        attr_map = torch.sigmoid(self.token_score(tokens)).transpose(1, 2).reshape(
            tokens.shape[0], 1, hw[0], hw[1]
        )
        return attr_logits, attr_map


class ContextualSelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        hidden = max(channels // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        avg, mx = x.mean(1, keepdim=True), x.amax(1, keepdim=True)
        spatial = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        channel = self.channel(x)
        return x * spatial * channel, spatial, channel


class BivariateFusion(nn.Module):
    """Lightweight approximation of the paper's Bivariate Fusion Module (BFM)."""

    def __init__(self, local_ch: int, global_ch: int) -> None:
        super().__init__()
        self.local = nn.Conv2d(local_ch, local_ch, 1)
        self.global_ = nn.Conv2d(global_ch, local_ch, 1)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        hidden = max(local_ch // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(local_ch, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, local_ch, 1), nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        local_feat, global_feat = self.local(local_feat), self.global_(global_feat)
        fused = local_feat + global_feat
        spatial = torch.sigmoid(self.spatial(torch.cat([
            fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)
        ], dim=1)))
        channel = self.channel(fused)
        return fused * spatial * channel + local_feat, spatial, channel


class ExUTransLite(nn.Module):
    """Runnable research approximation, not the paper's undisclosed exact implementation."""

    def __init__(self, in_ch=4, out_ch=1, patch_size=16, embed_dim=128, num_transformer_layers=4):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)
        self.patch = PatchEmbed(in_ch, embed_dim, patch_size)
        self.transformer_blocks = nn.ModuleList([
            SEMHABlock(embed_dim) for _ in range(num_transformer_layers)
        ])
        self.dae = DAE(embed_dim)
        self.glob = nn.Conv2d(embed_dim, 256, 1)
        self.fuse = BivariateFusion(256, 256)
        self.ctx = ContextualSelfAttention(256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)
        tokens, hw = self.patch(x)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        attr_logits, attr_map = self.dae(tokens, hw)
        global_map = tokens.transpose(1, 2).reshape(x.shape[0], -1, hw[0], hw[1])
        global_map = F.interpolate(global_map, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        global_map = self.glob(global_map)
        fused, fusion_spatial, fusion_channel = self.fuse(bottleneck, global_map)
        fused, ctx_spatial, ctx_channel = self.ctx(fused)
        d3 = self.up3(fused, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        out = self.head(d1)
        last_attention = self.transformer_blocks[-1].attn_map if self.transformer_blocks else None
        return out, {
            "attr_logits": attr_logits,
            "attr_map": F.interpolate(attr_map, size=x.shape[-2:], mode="bilinear", align_corners=False),
            "attention_map": last_attention,
            "fusion_spatial": fusion_spatial,
            "fusion_channel": fusion_channel,
            "context_spatial": ctx_spatial,
            "context_channel": ctx_channel,
        }


def create_model(in_ch=4, out_ch=1, patch_size=16, device="cpu", num_transformer_layers=4):
    return ExUTransLite(
        in_ch=in_ch,
        out_ch=out_ch,
        patch_size=patch_size,
        num_transformer_layers=num_transformer_layers,
    ).to(device)
