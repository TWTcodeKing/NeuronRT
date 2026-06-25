"""Custom Spikformer (not in SpikingJelly).

A structurally faithful Spikformer: a Spiking Patch Splitting (SPS) conv stem + stacked
Spiking-Self-Attention (SSA) / spiking-MLP blocks + a linear head. Built from SpikingJelly's
`layer` (Conv2d/Linear/BatchNorm2d) and `neuron.LIFNode` + einops.

The compiler only needs the module GRAPH and weight shapes (conv / dense / attention), not the
temporal dynamics, so this is single-step (`step_mode='s'`); a dummy forward of [B,C,H,W] suffices
for shape inference. Hyperparameters are illustrative — what matters is exercising conv + dense +
self-attention connectivity for the compiler.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from spikingjelly.activation_based import layer, neuron


class SPS(nn.Module):
    """Spiking Patch Splitting: two stride-2 conv stages -> patch tokens [B, N, embed_dim]."""

    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        mid = embed_dim // 2
        self.c1 = layer.Conv2d(in_ch, mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(mid)
        self.lif1 = neuron.LIFNode()
        self.c2 = layer.Conv2d(mid, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(embed_dim)
        self.lif2 = neuron.LIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W] -> [B, N, embed_dim]
        x = self.lif1(self.bn1(self.c1(x)))
        x = self.lif2(self.bn2(self.c2(x)))
        return rearrange(x, "b c h w -> b (h w) c")


class SSA(nn.Module):
    """Spiking Self-Attention (spike-driven, no softmax)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = layer.Linear(dim, dim, bias=False)
        self.k = layer.Linear(dim, dim, bias=False)
        self.v = layer.Linear(dim, dim, bias=False)
        self.q_lif = neuron.LIFNode()
        self.k_lif = neuron.LIFNode()
        self.v_lif = neuron.LIFNode()
        self.proj = layer.Linear(dim, dim, bias=False)
        self.proj_lif = neuron.LIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, dim]
        h = self.heads
        q = rearrange(self.q_lif(self.q(x)), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.k_lif(self.k(x)), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.v_lif(self.v(x)), "b n (h d) -> b h n d", h=h)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        out = attn @ v  # [B, h, N, d]
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj_lif(self.proj(out))


class SpikingMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = layer.Linear(dim, hidden, bias=False)
        self.lif1 = neuron.LIFNode()
        self.fc2 = layer.Linear(hidden, dim, bias=False)
        self.lif2 = neuron.LIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif2(self.fc2(self.lif1(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.attn = SSA(dim, heads)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)  # SEW-style additive residual
        x = x + self.mlp(x)
        return x


class Spikformer(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3, embed_dim: int = 256,
                 depth: int = 2, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.sps = SPS(in_ch, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.head = layer.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W] -> [B, num_classes]
        x = self.sps(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)  # global average over tokens
        return self.head(x)


def spikformer(num_classes: int = 10, embed_dim: int = 256, depth: int = 2,
               heads: int = 8) -> nn.Module:
    return Spikformer(num_classes=num_classes, embed_dim=embed_dim, depth=depth, heads=heads)
