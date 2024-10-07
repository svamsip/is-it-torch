"""Modules for attention mechanism: Scaled Dot-Product Attention, Multi-Head Attention, Position-wise Feed-Forward Network"""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism
    params:
        d_model: int, dimension of model - embedding_size
        d_k: int, dimension of K/Q
        d_v: int, dimension of V
    """

    def __init__(self):
        super().__init__()
        self.d_k = None

    def forward(self, Q, K, V, mask=None):

        self.d_k = K.size(-1)
        # Scaled dot-product attention
        # coefficients for V (z): (Q.K^T)/sqrt(d_k)
        # sqrt(d_k) to scale the dot product

        # Reshape Q, K, V to 3D tensors for bmm
        batch_size, num_heads, seq_len, d_k = Q.size()
        Q = Q.reshape(batch_size * num_heads, seq_len, d_k)
        K = K.reshape(batch_size * num_heads, seq_len, d_k)
        V = V.reshape(batch_size * num_heads, seq_len, d_k)

        z_scores = torch.bmm(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            mask = mask.repeat(num_heads, 1, 1)
            z_scores = z_scores.masked_fill(mask == 0, -1e9)
        attn = nn.Softmax(dim=-1)(z_scores)
        context = torch.bmm(attn, V)

        # Reshape context back to 4D tensor
        context = context.view(batch_size, num_heads, seq_len, d_k)

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    params:
        d_model: int, dimension of model
        # d_k: int, dimension of K/Q
        # d_v: int, dimension of V
        h: int, number of heads in multi-head attention

    """

    def __init__(self, d_model, h):
        super().__init__()
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        # weight matrices:
        self.W_Q = nn.Linear(
            in_features=d_model,
            out_features=h * self.d_k,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )

        self.W_K = nn.Linear(
            in_features=d_model,
            out_features=h * self.d_k,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )

        self.W_V = nn.Linear(
            in_features=d_model,
            out_features=h * self.d_v,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )

        # final linear layer: concatentation of all heads
        self.fc = nn.Linear(
            in_features=h * self.d_v,
            out_features=d_model,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )

        self.scaled_dot_product_attention_layer = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):

        # Linear transformation
        q = self.W_Q(Q)  # (batch_size, seq_len, h * d_k)
        k = self.W_K(K)  # (batch_size, seq_len, h * d_k)
        v = self.W_V(V)  # (batch_size, seq_len, h * d_v)

        # transform into multiple heads
        batch_size = q.size(0)
        seq_len = q.size(1)

        # (batch_size, seq_len, h * d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.h, self.d_v).transpose(1, 2)

        # Scaled Dot-Product Attention: context (batch_size, h, seq_len, d_v)
        context, attn = self.scaled_dot_product_attention_layer(q, k, v, mask)

        # concatenate heads: (batch_size, h, seq_len, d_v) -> (batch_size, seq_len, h, d_v) -> (batch_size, seq_len, h * d_v)
        ## ! tensor.contiguous()
        context = context.transpose(1, 2).reshape(
            batch_size, seq_len, self.h * self.d_v
        )

        # final linear layer: output (batch_size, seq_len, d_model)
        output = self.fc(context)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network
    params:
        d_model: int, dimension of model
        d_ff: int, dimension of feed-forward layer
        dropout: float, dropout rate
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
