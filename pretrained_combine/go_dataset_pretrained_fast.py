import os
import pickle
from torch.utils.data import Dataset
import collections
import h5py
import numpy as np
import torch


class GoDataset(Dataset):
    '''Class for go dataset'''

    def __init__(self, config, split='train'):
        super(GoDataset, self).__init__()
        self.split = split
        self.data_dir = config['data_dir']
        self.device = config['device']
        self.portion = config['portion']

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
        h5_path = os.path.join(self.data_dir, self.split + '_board_inputs.h5')

        with h5py.File(h5_path, 'r') as hf:
            bin_input_datas = hf.get('bin_input_datas')
            global_input_datas = hf.get('global_input_datas')
            self.data_raw['bin_input_datas'] = np.array(bin_input_datas) # 'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas
            self.data_raw['global_input_datas'] = np.array(global_input_datas)
            print('Boards shape', self.data_raw['bin_input_datas'].shape)

    def get_text(self):
        print("------ Loading text ------")
        h5_path = os.path.join(self.data_dir, self.split + '_text_inputs.h5')
        with h5py.File(h5_path, 'r') as hf:
            comments = np.array(hf.get('comments'))
            vocab_size = int(np.array(hf.get('vocab_size')))
            hf.close()
            print("vocab_size", vocab_size)

            self.data_raw['texts'] = torch.tensor(comments).to(self.device)
            self.vocab_size = vocab_size
            print('Texts shape', self.data_raw['texts'].shape)

    def get_choices(self):
        print("------ Loading choices ------")
        choices_path = os.path.join(self.data_dir, f'{self.split}.choices.pkl')
        with open(choices_path, 'rb') as input_file:
            self.choices = pickle.load(input_file)

    def get_example(self, board_idx, text_idx, label):
        bin_input_datas = self.data_raw['bin_input_datas'][board_idx]
        global_input_datas = self.data_raw['global_input_datas'][board_idx]
        text = self.data_raw['texts'][text_idx]
        return {'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas,
                'text': text.to(self.device), 'label': label}

    def get_pos_neg_examples(self):
        print("------ Loading positive and negative examples ------")
        for index in range(int(self.portion * len(self.choices['choice_indices']))):
            choice_indices = self.choices['choice_indices'][index]
            pos_idx = choice_indices[self.choices['answers'][index]]
            neg_idx = choice_indices[1] if self.choices['answers'][index] == 0 else choice_indices[0]

            pos_example = self.get_example(pos_idx, pos_idx, 1)
            neg_example = self.get_example(pos_idx, neg_idx, 0)
            self.data.append(pos_example)
            self.data.append(neg_example)
