import torch
import math


def positional_encoding(seq_len, d_model):
    """
    Create positional encoding for a sequence with length 'seq_len' and model dimension 'd_model'.

    :param seq_len: Length of the input sequence.
    :param d_model: Dimension of the embeddings/model.
    :return: Positional encoding matrix of shape [seq_len, d_model].
    """
    # Initialize the positional encoding matrix
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -
                         (math.log(10000.0) / d_model))

    # Compute the positional encodings
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
