import torch
import torch.nn as nn
from attention import MultiHeadAttention, PositionwiseFeedForward


class Decoder(nn.Module):
    """
    Decoder class
    """

    def __init__(self, d_model, num_stacks=6):
        super().__init__()
        self.num_stacks = num_stacks
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, h=8)
        self.cross_attention = MultiHeadAttention(d_model=d_model, h=8)
        self.positionwise_feedforward = PositionwiseFeedForward(
            d_model=d_model, d_ff=2048, dropout=0.2
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):
        for _ in range(self.num_stacks):
            x = self.layer_norm1(x + self.multi_head_attention(x, x, x, mask)[0])
            x = self.layer_norm2(
                x + self.cross_attention(x, encoder_output, encoder_output, None)[0]
            )
            x = self.layer_norm3(x + self.positionwise_feedforward(x))
        return x
