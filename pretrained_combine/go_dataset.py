import os
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset
import collections

PAD = '<PAD>'


class GoDataset(Dataset):
    '''Class for go dataset'''

    DIR = 'data_splits_final'

    def __init__(self, config, split='train'):
        self.data = []
        self.split = split
        self.data_raw = {}
        self.id2tok = []
        self.tok2id = {}
        self.choices = {}

        self.get_board()
        self.get_text()
        self.get_choices()

    def __getitem__(self, index):
        ''' Get the positive and negative examples at index '''
        if isinstance(index, collections.Sequence):
            return tuple(
                (i, self.get_pos_neg_examples(i)) for i in index
            )
        else:
            return index, self.get_pos_neg_examples(index)

    def get_board(self):
        with open(os.path.join(GoDataset.DIR, self.split + '.pkl'), 'rb') as input_file:
            board_data = pickle.load(input_file)
        self.data_raw['row'] = [board[0] for board in board_data['boards']]
        self.data_raw['col'] = [board[1] for board in board_data['boards']]
        self.data_raw['color'] = [board[2] for board in board_data['boards']]
        self.data_raw['boards'] = [board[3] for board in board_data['boards']]

    def get_text(self):
        tensorized_path = os.path.join(GoDataset.DIR, self.split + '_comments.tok.32000.bin')
        if not os.path.exists(tensorized_path):
            with open(os.path.join(GoDataset.DIR, self.split + '_comments.tok.32000.txt'), 'rt') as input_file:
                text_data = [line.strip().split() for line in input_file.readlines()]

            # get vocab
            with open(os.path.join(GoDataset.DIR, 'vocab.32000'), 'rt') as input_file:
                self.id2tok = [line.strip().split()[0] for line in input_file.readlines()]
                for id, tok in enumerate(iter(self.id2tok)):
                    self.tok2id[tok] = id
            self.tok2id[PAD] = len(self.id2tok)
            self.id2tok.append(PAD)

            text_data_idxs = []
            # tensorize text
            for line in text_data:
                text_data_idxs.append([self.tok2id[tok] for tok in line])

            self.data_raw['text'] = text_data_idxs

            with open(tensorized_path, 'wb') as output_file:
                pickle.dump(text_data_idxs, output_file)

        else:
            with open(tensorized_path, 'rb') as input_file:
                self.data_raw['text'] = pickle.load(input_file)

    def get_choices(self):
        choices_path = os.path.join(GoDataset.DIR, self.split + '.choices.pkl')
        with open(choices_path, 'wb') as input_file:
            self.choices = pickle.load(input_file)

    @property
    def padding_idx(self):
        return self.tok2id[PAD]

    def collate_field(self, values):
        ''' Collate a specific field '''
        batch = nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=self.padding_idx)
        batch_lens = torch.LongTensor([len(sequence) for sequence in values])
        return batch, batch_lens

    def get_pos_neg_examples(self, index):
        choice_indices = self.choices['choice_indices'][index]
        answer_idx = choice_indices[self.choices['answer'][index]]
        row = self.data_raw['row'][answer_idx]
        col = self.data_raw['col'][answer_idx]
        color = self.data_raw['color'][answer_idx]
        board = self.data_raw['board'][answer_idx]
        text = [torch.LongTensor(self.data_raw['text'][i]) for i in choice_indices]
        return row, col, color, board, text
