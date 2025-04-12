import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformer import Transformer

writer = SummaryWriter("runs/transformer-architecture")


def train_transformer(vocab_size, d_model, max_seq_len):
    """
    Train Transformer model
    """
    # instantiate model
    model = Transformer(vocab_size, d_model, max_seq_len)
    # dummy input
    batch_size = 8
    inputs = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    outputs = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    print("batch_size: ", batch_size)
    print("inputs.shape: ", inputs.shape)
    print("outputs.shape: ", outputs.shape)

    print("model: ", model)
    print("--" * 40)
    print("model: ", Transformer(vocab_size=10000, d_model=512, max_seq_len=512))
    writer.add_graph(model=model, input_to_model=(inputs, outputs), verbose=False)
    writer.close()
    # forward pass
    preds = model(inputs, outputs)

    print("preds.shape: ", preds.shape)


if __name__ == "__main__":
    train_transformer(vocab_size=10000, d_model=512, max_seq_len=512)
