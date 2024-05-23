import math

import torch

from torch import nn


class PositionalEncoding(nn.Module):
    """Взял отсюда: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute((1, 0, 2))
        x = x + self.pe[:x.size(0)]
        return self.dropout.forward(x).permute((1, 0, 2))


class RecModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_head: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head

        self.pos_encoder = PositionalEncoding(self.in_dim)
        encoder_layer = nn.TransformerEncoderLayer(self.in_dim, nhead=self.n_head, batch_first=True)
        self.encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.linear = nn.Linear(self.in_dim, out_dim)
        # self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x *= math.sqrt(self.in_dim)

        x = self.pos_encoder(x)
        x = self.encoder_1(x)
        x = self.linear(x)

        x = x.mean(dim=1)

        return x
