import torch
from torch import nn


class TextEncoder(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", num_encoder_layers: int = 6):
        super(TextEncoder, self).__init__()
        embedding = nn.Embedding()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, src):
        return self.encoder(src)