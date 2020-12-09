import sys
sys.path.append('.')
sys.path.append('..')

import os
import h5py
from transformer_encoder import data_process

data_dir = 'data_splits_final'
splits = ['train', 'val', 'test']


for split in splits:
    print(f'----{split}----')

    comments_filepath = os.path.join(data_dir, f'{split}_comments.tok.32000.txt')
    vocab_filepath = os.path.join(data_dir, 'vocab.32000')
    print('reading comments...')
    comments, vocab_size = data_process.read_comment_subword(comments_filepath, vocab_filepath, 5)

    h5_path_output = os.path.join(data_dir, f'{split}_text_inputs.h5')

    hf = h5py.File(h5_path_output, 'w')
    print('saving to h5...')
    hf.create_dataset('comments', data=comments)
    hf.create_dataset('vocab_size', data=vocab_size)
    hf.close()
    # with open(h5_path_output, 'wb') as output_file:
    #     pickle.dump({'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas}, output_file)