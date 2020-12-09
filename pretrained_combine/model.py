import torch
from torch import nn


class PretrainedCombineModel(nn.Module):

    def __init__(self, combine='concat', d_model=512, dropout_p=0.1, sentence_len=100, text_hidden_dim=200, board_embed_size=256,
                 num_heads=4):
        super(PretrainedCombineModel, self).__init__()
        self.combine = combine
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.text_hidden_size = sentence_len
        self.text_embed_size = text_hidden_dim
        self.board_size = 19
        self.board_embed_size = board_embed_size
        self.num_heads = num_heads

        if combine == 'concat':
            self.fc = nn.Linear(self.board_size * self.board_size * self.board_embed_size
                                + self.text_hidden_size * self.text_embed_size,
                                1)
        elif combine == 'dot':
            self.fc_board = nn.Linear(self.board_size * self.board_size * self.board_embed_size, self.d_model)
            self.fc_text = nn.Linear(self.text_hidden_size * self.text_embed_size, self.d_model)
        elif combine == 'attn':
            self.fc_board = nn.Linear(self.board_embed_size, self.d_model)
            self.fc_text = nn.Linear(self.text_embed_size, self.d_model)
            self.attn = nn.MultiheadAttention(self.d_model, self.num_heads, self.dropout_p)
        else:
            raise ValueError('Unrecognized combine type')

    def forward(self, board_embedding, text_embedding):
        """

        :param board_embedding: (batch_size, 19, 19, 128)
        :param text_embedding: (batch_size, self.text_hidden_size, self.text_embed_size)
        :return:
        """
        batch_size = board_embedding.size(0)
        if self.combine == 'concat':
            cat_embeddings = torch.cat((board_embedding.view(batch_size, -1), text_embedding.view(batch_size, -1)), dim=1)
            logits = self.fc(cat_embeddings)
        elif self.combine == 'dot':
            board_input = self.fc_board(board_embedding.view(batch_size, -1))[:, None, :]  # (batch_size,d_model) -> (batch_size, 1, d_model)
            text_input = self.fc_text(text_embedding.view(batch_size, -1))[:, :, None]  # (batch_size, d_model)  -> (batch_size, d_model, 1)
            logits = torch.bmm(board_input, text_input).view(-1, 1)
        elif self.combine == 'attn':
            board_input = self.fc_board(board_embedding).view(batch_size, -1, self.d_model).transpose(0, 1)
            text_input = self.fc_text(text_embedding).transpose(0, 1)
            attn_output, attn_output_weights = self.attn(text_input, board_input, board_input)
            logits = torch.mean(attn_output, dim=[0, 2])[:, None]
        else:
            raise ValueError('Unrecognized combine type')

        return logits
