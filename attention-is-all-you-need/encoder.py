import torch
import torch.nn as nn
from attention import MultiHeadAttention, PositionwiseFeedForward


class Encoder(nn.Module):
    """
    Encoder class
    """

    def __init__(self, d_model=512, num_stacks=6):
        super().__init__()
        self.num_stacks = num_stacks
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, h=8)
        self.positionwise_feedforward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=2048,
            dropout=0.2,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        for _ in range(self.num_stacks):
            x = self.layer_norm1(x + self.multi_head_attention(x, x, x)[0])
            x = self.layer_norm2(x + self.positionwise_feedforward(x))
        return x
