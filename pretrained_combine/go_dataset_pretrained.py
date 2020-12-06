import os
import pickle
from torch.utils.data import Dataset
import collections
from transformer_encoder.model import *
from transformer_encoder import data_process


class GoDataset(Dataset):
    '''Class for go dataset'''

    def __init__(self, config, split='train'):
        super(GoDataset, self).__init__()
        self.split = split
        self.data_dir = config['data_dir']
        self.device = config['device']

        self.data = []
        self.vocab_size = None
        self.data_raw = {}
        self.choices = {}

        self.get_board()
        self.get_text()
        self.get_choices()
        self.get_pos_neg_examples()

    def __getitem__(self, index):
        ''' Get the positive and negative examples at index '''
        if isinstance(index, collections.Sequence):
            return tuple(
                (i, self.data[i]) for i in index
            )
        else:
            return index, self.data[index]

    def __len__(self):
        return len(self.data)

    def get_board(self):
        print("------ Loading boards ------")
        pkl_path = os.path.join(self.data_dir, self.split + '.pkl')

        with open(pkl_path, 'rb') as input_file:
            board_data = pickle.load(input_file)
        self.data_raw['rows'] = np.array([board[0] for board in board_data['boards']])
        self.data_raw['cols'] = np.array([board[1] for board in board_data['boards']])
        self.data_raw['colors'] = np.array([board[2] for board in board_data['boards']])
        self.data_raw['boards'] = np.array([board[3] for board in board_data['boards']])
        print('Boards shape', self.data_raw['boards'].shape)

    def get_text(self):
        print("------ Loading text ------")
        comments_filepath = os.path.join(self.data_dir,  f'{self.split}_comments.tok.32000.txt')
        vocab_filepath = os.path.join(self.data_dir,  'vocab.32000')
        comments, vocab_size = torch.tensor(data_process.read_comment_subword(comments_filepath, vocab_filepath, 5))

        self.data_raw['texts'] = comments
        self.vocab_size = vocab_size
        print('Texts shape', self.data_raw['texts'].shape)

    def get_choices(self):
        print("------ Loading choices ------")
        choices_path = os.path.join(self.data_dir, f'{self.split}.choices.pkl')
        with open(choices_path, 'rb') as input_file:
            self.choices = pickle.load(input_file)

    def get_example(self, board_idx, text_idx, label):
        row = self.data_raw['rows'][board_idx]
        col = self.data_raw['cols'][board_idx]
        color = self.data_raw['colors'][board_idx]
        board = self.data_raw['boards'][board_idx]
        text = self.data_raw['texts'][text_idx]
        return {'row': row, 'col': col, 'color': color, 'board': board, 'text': text.to(self.device), 'label': label}

    def get_pos_neg_examples(self):
        print("------ Loading positive and negative examples ------")
        for index in range(len(self.choices['choice_indices'])):
            choice_indices = self.choices['choice_indices'][index]
            pos_idx = choice_indices[self.choices['answers'][index]]
            neg_idx = choice_indices[1] if self.choices['answers'][index] == 0 else choice_indices[0]

            pos_example = self.get_example(pos_idx, pos_idx, 1)
            neg_example = self.get_example(pos_idx, neg_idx, 0)
            self.data.append(pos_example)
            self.data.append(neg_example)
