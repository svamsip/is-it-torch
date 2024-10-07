import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from utils import Embedding, PositionalEncoding


class Transformer(nn.Module):
    """
    Transformer class
    """

    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder = Encoder(d_model=d_model, num_stacks=6)
        self.decoder = Decoder(d_model=d_model, num_stacks=6)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, outputs, decoder_mask=None):

        # Embedding
        input_emb = self.embedding(inputs)
        output_emb = self.embedding(outputs)

        # Positional Encoding
        encode_input = input_emb + self.positional_encoding(input_emb)
        decoder_input = output_emb + self.positional_encoding(output_emb)

        # Encoder
        encoder_output = self.encoder(encode_input)

        # Decoder
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask)

        x = self.linear(decoder_output)
        x = self.softmax(x)
        return x
