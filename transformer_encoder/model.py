import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

MAX_LENGTH = 100

class TransformerModel(nn.Module):

  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    super(TransformerModel, self).__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncoding(ninp, dropout)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.encoder = nn.Embedding(ntoken, ninp)
    self.ninp = ninp
    self.fc1 = nn.Linear(ninp, ninp*2)
    self.fc2 = nn.Linear(ninp*2, ntoken)
    self.dropout = nn.Dropout(p=0.5)

    self.init_weights()

  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.fc1.bias.data.zero_()
    self.fc1.weight.data.uniform_(-initrange, initrange)
    self.fc2.bias.data.zero_()
    self.fc2.weight.data.uniform_(-initrange, initrange)
  def forward(self, src, src_mask):
    src = self.encoder(src) 
    src = src * math.sqrt(self.ninp)
    src = self.pos_encoder(src)
    feature = self.transformer_encoder(src, src_mask) #features from encoder
    output = self.dropout(F.relu(self.fc1(feature)))
    output = self.fc2(output)
    return F.log_softmax(output, dim=-1)

class TransformerModel_extractFeature(nn.Module):

  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    super(TransformerModel_extractFeature, self).__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncoding(ninp, dropout)
    encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.encoder = nn.Embedding(ntoken, ninp)
    self.ninp = ninp
    #self.decoder = nn.Linear(ninp, ntoken)
    self.fc1 = nn.Linear(ninp, ninp*2)
    self.fc2 = nn.Linear(ninp*2, ntoken)
    self.dropout = nn.Dropout(p=0.5)


    self.init_weights()

  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.fc1.bias.data.zero_()
    self.fc1.weight.data.uniform_(-initrange, initrange)
    self.fc2.bias.data.zero_()
    self.fc2.weight.data.uniform_(-initrange, initrange)


  def forward(self, src, src_mask):
    src = self.encoder(src)
    src = src * math.sqrt(self.ninp)
    src = self.pos_encoder(src)
    feature = self.transformer_encoder(src, src_mask) #features from encoder
    output = self.dropout(F.relu(self.fc1(feature)))
    output = self.fc2(output)

    #output = self.transformer_encoder(src, src_mask)
    #output1 = self.decoder(output)
    return feature,output



class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=MAX_LENGTH):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)



