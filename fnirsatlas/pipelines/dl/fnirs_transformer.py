# Adapted from https://github.com/wzhlearning/fNIRS-Transformer (MIT, Copyright (c) 2021 wzhlearning).

"""Dual-branch fNIRS-T based on Zenghui Wang's reference implementation.

Reference: https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
"""

from typing import cast

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.nn import functional as F


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        return self.fn(X, **kwargs) + X


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, X, **kwargs):
        return self.fn(self.norm(X), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)


class _Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Identity()
            if heads == 1 and dim_head == dim
            else nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        )

    def forward(self, X, mask=None):
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in self.to_qkv(X).chunk(3, dim=-1)
        )
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            if mask.shape[-1] != dots.shape[-1]:
                raise ValueError("mask has incorrect dimensions")
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)
        attn = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _Residual(
                            _PreNorm(dim, _Attention(dim, heads, dim_head, dropout))
                        ),
                        _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim, dropout))),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, X, mask=None):
        for layer in self.layers:
            attention, feed_forward = cast(nn.ModuleList, layer)
            X = attention(X, mask=mask)
            X = feed_forward(X)
        return X


def _embedding(n_chromophores, height, n_times, dim):
    return nn.Sequential(
        nn.Conv2d(n_chromophores, 8, kernel_size=(height, 30), stride=(1, 4)),
        Rearrange("b c h w -> b h (c w)"),
        nn.Linear(((n_times - 30) // 4 + 1) * 8, dim),
        nn.LayerNorm(dim),
    )


class fNIRS_T(nn.Module):
    """Classify epochs using patch-wise and channel-wise transformer branches.

    Parameters
    ----------
    n_channels : int
        Number of channels per chromophore.
    n_times : int
        Epoch length, at least 30 samples.
    n_outputs : int
        Number of output classes.
    n_chromophores : int, default=2
        Number of chromophore planes.
    dim : int, default=64
        Token embedding dimension.
    depth : int, default=6
        Transformer blocks per branch.
    heads : int, default=8
        Attention heads per block.
    mlp_dim : int, default=64
        Feed-forward hidden dimension.
    pool : {'cls', 'mean'}, default='cls'
        Pooling over the class token or all tokens, including the class token.
    dim_head : int, default=64
        Dimension of each attention head.
    dropout : float, default=0.0
        Attention-output and feed-forward dropout probability.
    emb_dropout : float, default=0.0
        Token embedding dropout probability.
    """

    def __init__(
        self,
        n_channels,
        n_times,
        n_outputs,
        n_chromophores=2,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=64,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        if n_times < 30:
            raise ValueError("fNIRS-T requires at least 30 timepoints")
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")
        # These dimensions correspond to n_chs, sampling_point, n_class, and
        # n_ch_types in the reference and are inferred by FNIRSClassifier.
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.n_chromophores = n_chromophores
        self.pool = pool
        # A full-height kernel retains one patch token for montages smaller
        # than the reference's five-channel spatial kernel.
        patch_height = min(5, n_channels)
        self.to_patch_embedding = _embedding(n_chromophores, patch_height, n_times, dim)
        self.to_channel_embedding = _embedding(n_chromophores, 1, n_times, dim)
        # Allocate only the positions each branch uses, plus its class token.
        # The reference allocates 100 positions and slices them during forward.
        n_patches = n_channels - patch_height + 1
        self.pos_embedding_patch = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = _Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, n_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = _Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, n_outputs)
        )

    def _encode(self, X, embedding, token, positions, dropout, transformer, mask):
        X = embedding(X)
        cls_tokens = repeat(token, "() n d -> b n d", b=X.shape[0])
        X = torch.cat((cls_tokens, X), dim=1)
        X = transformer(dropout(X + positions), mask=mask)
        return X.mean(dim=1) if self.pool == "mean" else X[:, 0]

    def forward(self, X, mask=None):
        """Return logits for chromophore-split epochs.

        Parameters
        ----------
        X : torch.Tensor
            Input shaped (batch, chromophores, channels, timepoints).
        mask : torch.Tensor or None
            Boolean validity mask shared by both branches, excluding class
            tokens. Only applicable when both branches have equal token counts.

        Returns
        -------
        torch.Tensor
            Class logits shaped (batch, n_outputs).
        """
        patch = self._encode(
            X,
            self.to_patch_embedding,
            self.cls_token_patch,
            self.pos_embedding_patch,
            self.dropout_patch,
            self.transformer_patch,
            mask,
        )
        channel = self._encode(
            X,
            self.to_channel_embedding,
            self.cls_token_channel,
            self.pos_embedding_channel,
            self.dropout_channel,
            self.transformer_channel,
            mask,
        )
        return self.mlp_head(torch.cat((patch, channel), dim=1))
