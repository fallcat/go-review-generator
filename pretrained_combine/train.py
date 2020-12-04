import sys
sys.path.append('.')
sys.path.append('..')

import time

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

import numpy as np
import tensorflow as tf
import katago
from katago.board import IllegalMoveError

from pretrained_combine.go_dataset_pretrained import GoDataset
from torch.utils.data import DataLoader

from transformer_encoder import get_comment_features
from transformer_encoder.model import *

from pretrained_combine.model import PretrainedCombineModel
from pretrained_combine.utils import WarmupLRSchedule

torch.manual_seed(42)

# configs
data_dir = 'data_splits_final'
# katago_model_dir = 'katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/'
katago_model_dir = 'katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/'

config = {'data_dir': data_dir}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_epoch = 10

# text config
emsize = 200 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
# combine config
combine = 'concat'
d_model = 512
dropout_p = 0.1
board_embed_size = 128
# optim conifg
learning_rate = 3e-4
scheduler_type = 'warmup'
warmup_steps = 4000
# Dataloader
train_set = GoDataset(config, split='val')

dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Load pretrained katago model
board_model, model_variables_prefix, model_config_json = katago.get_model(katago_model_dir)
saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
)

# Load pretrained text Tranformer model
ntokens = train_set.vocab_size # the size of vocabulary

text_model = TransformerModel_extractFeature(ntokens, emsize, nhead, nhid, nlayers, dropout)
if torch.cuda.is_available():
    text_model = text_model.cuda()
for param in text_model.parameters():
    param.requires_grad = False  # freeze params in the pretrained model

# Construct the model
combine_model = PretrainedCombineModel(combine=combine,d_model=d_model, dropout_p=dropout_p, nhid=nhid, emsize=emsize, board_embed_size=board_embed_size)
if torch.cuda.is_available():
    combine_model = combine_model.cuda()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(combine_model.parameters(), learning_rate)
if scheduler_type == 'warmup':
    lr_scheduler = LambdaLR(
        optimizer,
        WarmupLRSchedule(
            warmup_steps
        )
    )
else:
    raise ValueError('Unknown scheduler type: ', scheduler_type)


with tf.Session() as session:
    saver.restore(session, model_variables_prefix)

    num_batches = len(train_set) / batch_size
    epochs = tqdm(range(num_epoch))
    for epoch in epochs:
        epochs.set_description(f'Epoch {epoch}')
        batches = tqdm(enumerate(dataloader))
        for i_batch, sampled_batched in batches:
            batches.set_description(f'batch {i_batch}/{num_batches}')
            row = sampled_batched[1]['row'].numpy()
            col = sampled_batched[1]['col'].numpy()
            color = sampled_batched[1]['color']
            board = sampled_batched[1]['board'].numpy()
            text = sampled_batched[1]['text']
            label = sampled_batched[1]['label'][:, None]

            # model
            start = time.time()
            # features1 = katago.extract_intermediate_optimized.extract_features_batch(session, board_model, board[0], color[0])
            # print("time board1 features", time.time() - start)
            # start = time.time()
            try:
                board_features = torch.tensor(katago.extract_intermediate_optimized.extract_features_batch(session, board_model, board, color)).to(device)
            except IllegalMoveError:
                print(f"IllegalMoveError, skipped batch {sampled_batched[0]}")
                continue
            print("time board features", time.time() - start)
            start = time.time()
            text_features = torch.tensor(get_comment_features.extract_comment_features(text_model, text.to(device), batch_size, device)).to(device)

            print("time text features", time.time() - start)
            start = time.time()
            logits = combine_model(board_features, text_features)
            loss = criterion(logits, label.type_as(logits))
            print("time model", time.time() - start)
            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
