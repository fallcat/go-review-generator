import torch
from torch import nn


class PretrainedCombineModel(nn.Module):

    def __init__(self, combine='concat', d_model=512, dropout_p=0.1, nhid=100, emsize=200, board_embed_size=256):
        super(PretrainedCombineModel, self).__init__()
        self.combine = combine
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.text_hidden_size = nhid
        self.text_embed_size = emsize
        self.board_size = 19
        self.board_embed_size = board_embed_size

        if combine == 'concat':
            self.fc = nn.Linear(self.board_size * self.board_size * self.board_embed_size
                                + self.text_hidden_size * self.text_embed_size,
                                1)
        else:
            raise ValueError('Unrecognized combine type')

    def forward(self, board_embedding, text_embedding):
        """

        :param board_embedding: (batch_size, 19, 19, 256)
        :param text_embedding: (batch_size, self.text_hidden_size, self.text_embed_size)
        :return:
        """
        batch_size = board_embedding.size(0)
        if self.combine == 'concat':
            cat_embeddings = torch.cat((board_embedding.view(batch_size, -1), text_embedding.view(batch_size, -1)), dim=1)
            logits = self.fc(cat_embeddings)
            return logits
        else:
            raise ValueError('Unrecognized combine type')
