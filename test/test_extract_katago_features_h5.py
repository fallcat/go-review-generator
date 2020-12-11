import sys
sys.path.append('.')
sys.path.append('..')

import torch
import katago
import tensorflow as tf

import os
import h5py
import numpy as np

data_dir = 'data_splits_final'
split = 'val'

h5_path = os.path.join(data_dir, split + '_board_inputs.h5')

with h5py.File(h5_path, 'r') as hf:
    bin_input_datas = hf.get('bin_input_datas')
    global_input_datas = hf.get('global_input_datas')
    bin_input_datas = np.array(bin_input_datas) # 'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas
    global_input_datas = np.array(global_input_datas)
    print('Boards shape', bin_input_datas.shape)

saved_model_dir = "katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/"
board_model, model_variables_prefix, model_config_json = katago.get_model(saved_model_dir)

saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with tf.Session() as session:
    saver.restore(session, model_variables_prefix)

    # bin_input_datas = ... # from some dataloader
    # global_input_datas = ... # from some dataloader

    board_features = katago.extract_intermediate.fetch_output_batch_with_bin_input(session, board_model, bin_input_datas[:10], global_input_datas[:10])
    print('board_features', type(board_features))