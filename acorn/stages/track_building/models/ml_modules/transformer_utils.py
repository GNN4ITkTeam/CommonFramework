# 3rd party imports
import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import Optional


class SetNorm(nn.Module):
    """
    Implemented SetNorm in `https://arxiv.org/abs/1810.00825`
    """

    def __init__(self, normalized_shape: int):
        """
        Initialize a setnorm instance
        argument:
            normalized_shape: the shape of inputs
        """
        super().__init__()
        self.weight = Parameter(torch.ones((1, 1, normalized_shape)))
        self.bias = Parameter(torch.zeros((1, 1, normalized_shape)))

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        """
        Normalize a batched input of shape (P, N, C), where P is the number
        of particles (sequence length), N is the batched dimension, and C is
        the normalized_shape
        arguments:
            x: a tensor of shape (P, N, C)
            mask: a tensor of shape (N, P)
        return
            normalized inputs of shape (P, N, C)
        """
        if mask is None:
            mask = torch.zeros((x.shape[1], x.shape[0]), device=x.device, dtype=bool)
        weights = (
            ((~mask).float() / (~mask).sum(1, keepdim=True)).permute(1, 0).unsqueeze(2)
        )
        means = (x * weights).sum(0, keepdim=True).mean(2, keepdim=True)  # [1, N, 1]
        variances = (
            ((x - means).square() * weights).sum(0, keepdim=True).mean(2, keepdim=True)
        )  # [1, N, 1]
        std_div = torch.sqrt(variances + 1e-5)  # [1, N, 1]
        return ((x - means) / std_div * self.weight + self.bias).masked_fill_(
            mask.permute(1, 0).unsqueeze(2), 0
        )


class AttentionBlock(nn.Module):
    """
    An attention block implementation based on https://nn.labml.ai/transformers/models.html
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = 2048,
        heads: Optional[int] = 8,
        dropout: Optional[float] = 0,
        d_source: Optional[int] = 512,
        self_attn: Optional[bool] = True,
        cross_attn: Optional[bool] = False,
        activation: Optional[nn.Module] = nn.GELU(),
    ):
        """
        Initialize an `AttentionBlock` instance
        arguments:
            d_model: the size of input
            d_ff: hidden size of the feed forward network
            heads: number of heads used in MHA
            dropout: dropout strength
            d_source: dimensionality of source if cross attention is used
            self_attn: whether to use self attention
            cross_attn: whether to use cross attention
            activation: activation function
        """
        super().__init__()
        if self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                d_model, heads, dropout=dropout, kdim=d_source, vdim=d_source
            )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout = nn.Dropout(dropout)
        if self_attn:
            self.norm_self_attn = SetNorm(d_model)
        if cross_attn:
            self.norm_cross_attn = SetNorm(d_model)
        self.norm_ff = SetNorm(d_model)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        src_padding_mask: torch.Tensor = None,
    ) -> torch.tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (P, N, C)
            src: input source sequence of shape (S, N, C')
            padding_mask: a mask of shape (P, N) with `True` represents a
                a real particle
            src_padding_mask: a mask of shape (S, N) with `True` represents a
                a real input
        returns:
            transformed sequence of shape (P, N, C)
        """
        if hasattr(self, "self_attn"):
            z = self.norm_self_attn(x, padding_mask)
            self_attn, *_ = self.self_attn(z, z, z, padding_mask)
            x = x + self.dropout(self_attn)
        if hasattr(self, "cross_attn"):
            z = self.norm_cross_attn(x, padding_mask)
            src_attn, *_ = self.cross_attn(z, src, src, src_padding_mask)
            x = x + self.dropout(src_attn)

        z = self.norm_ff(x, padding_mask)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x


class BTransformerBlock(nn.Module):
    """
    An attention block implementation based on https://nn.labml.ai/transformers/models.html
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = 2048,
        heads: Optional[int] = 8,
        dropout: Optional[float] = 0,
        activation: Optional[nn.Module] = nn.GELU(),
    ):
        super().__init__()
        self.hit_to_tracks = AttentionBlock(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            dropout=dropout,
            d_source=d_model,
            self_attn=True,
            cross_attn=True,
            activation=activation,
        )
        self.tracks_to_hits = AttentionBlock(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            dropout=dropout,
            d_source=d_model,
            self_attn=False,
            cross_attn=True,
            activation=activation,
        )

    def forward(self, hits, tracks):
        tracks = self.hit_to_tracks(tracks, hits)
        hits = self.tracks_to_hits(hits, tracks)
        return hits, tracks
