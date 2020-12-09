import sys
sys.path.append('.')
sys.path.append('..')

import os
import pickle
import numpy as np
from katago.extract_intermediate import get_model, extract_bin_input_batch

data_dir = 'data_splits_final'
splits = ['train', 'val', 'test']
saved_model_dir = "katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/"
model, model_variables_prefix, model_config_json = get_model(saved_model_dir)


for split in splits:

    pkl_path = os.path.join(data_dir, split + '.pkl')

    with open(pkl_path, 'rb') as input_file:
        board_data = pickle.load(input_file)
    rows = np.array([board[0] for board in board_data['boards']])
    cols = np.array([board[1] for board in board_data['boards']])
    colors = np.array([board[2] for board in board_data['boards']])
    boards = np.array([board[3] for board in board_data['boards']])
    print('Split', split, 'Boards shape', boards.shape)

    bin_input_datas, global_input_datas = extract_bin_input_batch(model, boards, colors, rows, cols, use_tqdm=True)

    pkl_path_output = os.path.join(data_dir, f'{split}_board_inputs.pkl')
    with open(pkl_path_output, 'wb') as output_file:
        pickle.dump({'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas}, output_file)