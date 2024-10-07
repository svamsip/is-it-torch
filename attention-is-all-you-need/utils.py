import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding class
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
        Even pos: PE(pos,2i) = sin(pos/10000^(2i/d_model))
        Odd pos: PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_enc = torch.zeros(max_seq_len, d_model)

        # get column vector for positional encoding: (max_seq_len, 1)
        pos_idx = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # denominator: [2i] aka even positions
        denominator = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # compute positional encoding
        self.pos_enc[:, 0::2] = torch.sin(pos_idx / denominator)
        self.pos_enc[:, 1::2] = torch.cos(pos_idx / denominator)

        # add batch dimension: (1, max_seq_len, d_model)
        self.pos_enc = self.pos_enc.unsqueeze(0)

    def forward(self, x):
        # add positional encoding to input
        x = x + self.pos_enc[:, : x.size(1), :].to(x.device)
        return x


def init_weights(m):
    """
    Initialize weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    return None


# TODO: Implement Byte Pair Encoding (BPE) algorithm
