import sys
sys.path.append('.')
sys.path.append('..')

import time
import pdb
import os
import shutil
import wandb

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
from pretrained_combine.utils import WarmupLRSchedule, checkpoint


torch.manual_seed(42)
wandb.init(project='go-review-matcher')

# configs
data_dir = 'data_splits_final'
# katago_model_dir = 'katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/'
katago_model_dir = 'katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/'
experiment_dir = 'experiments/exp01'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
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
# checkpoint config
checkpoint_interval = 5  # num steps
max_checkpoints = 5

# Dataloader
config = {'data_dir': data_dir,
          'katago_model_dir': katago_model_dir,
          'device': device,
          'experiment_dir': experiment_dir,
          'batch_size': batch_size,
          'num_epoch': num_epoch,
          }
train_set = GoDataset(config, split='train')
val_set = GoDataset(config, split='val')

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

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
wandb.watch(combine_model)
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



def evaluate(combine_model, board_model, text_model, val_dataloader, model_variables_prefix):
    with tf.Session() as session:
        saver.restore(session, model_variables_prefix)
        combine_model.eval()
        batches = tqdm(enumerate(val_dataloader))
        total_correct = 0
        total_loss = 0
        total_total = 0
        count = 0
        for i_batch, sampled_batched in batches:
            count += 1
            color = sampled_batched[1]['color']
            board = sampled_batched[1]['board'].numpy()
            text = sampled_batched[1]['text']
            label = sampled_batched[1]['label'][:, None]

            try:
                board_features = torch.tensor(
                    katago.extract_intermediate_optimized.extract_features_batch(session, board_model, board, color)).to(
                    device)
            except IllegalMoveError:
                print(f"IllegalMoveError, skipped batch {sampled_batched[0]}")
                continue

            text_features = torch.tensor(get_comment_features.extract_comment_features(text_model, text.to(device), batch_size, device)).to(device)
            logits = combine_model(board_features, text_features)
            loss = criterion(logits, label.type_as(logits))
            total_loss += loss
            pred = logits >= 0.5
            correct = sum(pred == label)
            total = label.shape[0]
            total_correct += correct
            total_total += total
            if count == 10:
                break
        accuracy = float(total_correct) / total_total
        loss_avg = float(total_loss) / total_total
    combine_model.train()
    return accuracy, loss_avg

num_batches = len(train_set) / batch_size
epochs = tqdm(range(num_epoch))
loss_accumulate = 0
count_accumulate = 0

modules = {'model_state_dict': combine_model,
           'optimizer_state_dict': optimizer,
           'lr_scheduler_dict': lr_scheduler}

val_loss_history = []

combine_model.train()
with tf.Session() as session:
    saver.restore(session, model_variables_prefix)
    step = 0
    for epoch in epochs:
        epochs.set_description(f'Epoch {epoch}')
        batches = tqdm(enumerate(train_dataloader))
        count = 0
        for i_batch, sampled_batched in batches:
            count += 1
            step += 1
            batches.set_description(f'batch {i_batch}/{num_batches}')
            row = sampled_batched[1]['row'].numpy()
            col = sampled_batched[1]['col'].numpy()
            color = sampled_batched[1]['color']
            board = sampled_batched[1]['board'].numpy()
            text = sampled_batched[1]['text']
            label = sampled_batched[1]['label'][:, None]

            try:
                board_features = torch.tensor(
                    katago.extract_intermediate_optimized.extract_features_batch(session, board_model, board, color)).to(
                    device)
            except IllegalMoveError:
                print(f"IllegalMoveError, skipped batch {sampled_batched[0]}")
                continue

            text_features = torch.tensor(get_comment_features.extract_comment_features(text_model, text.to(device), batch_size, device)).to(device)

            # print("time text features", time.time() - start)
            # start = time.time()
            logits = combine_model(board_features, text_features)
            loss = criterion(logits, label.type_as(logits))
            loss_accumulate += loss
            count_accumulate += sampled_batched[1]['label'].shape[0]
            # print("time model", time.time() - start)
            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if i_batch % checkpoint_interval == 0:
                checkpoint_path = checkpoint(epoch, i_batch, modules, experiment_dir, max_checkpoints=max_checkpoints)

                loss_avg = float(loss_accumulate) / count_accumulate
                val_acc, val_loss = evaluate(combine_model, board_model, text_model, val_dataloader, model_variables_prefix)
                val_loss_history.append(val_loss)
                if val_loss >= max(val_loss_history):  # best
                    dirname = os.path.dirname(checkpoint_path)
                    basename = os.path.basename(checkpoint_path)
                    best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
                    shutil.copy2(checkpoint_path, best_checkpoint_path)
                wandb.log({'Val Accuracy': val_acc,
                           'Val Loss': val_loss,
                           'Train Loss': loss_avg,
                           'lr': lr_scheduler.get_last_lr(),
                           'epoch': epoch,
                           'step': step})
                loss_accumulate = 0
                count_accumulate = 0

            if count == 10:
                break
