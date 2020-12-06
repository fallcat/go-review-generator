import sys
sys.path.append('.')
sys.path.append('..')

import time
import pdb
import os
import shutil
import wandb
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser()

    # paths and info
    parser.add_argument('--project-name', type=str, default='go-review-matcher', help='Project name')
    parser.add_argument('-d', '--data-dir', type=str, default='data_splits_final', help='Directory of data')
    parser.add_argument('--katago-model-dir', type=str,
                        default='katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/',
                        help='Directory of katago model')
    parser.add_argument('--experiment-dir', type=str, default='experiments/exp01',
                        help='Directory to save the experiment')

    # training config
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for each step')
    parser.add_argument('--num-epoch', type=int, default=10, help='Number of epochs to trian')
    parser.add_argument('--track', default=False, action='store_true', help='Use this flag to track the experiment')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Checkpoint every how many steps')
    parser.add_argument('--max-checkpoints', type=int, default=5, help='Max number of checkpoints to keep')
    parser.add_argument('--learning-rate', type=int, default=3e-4, help='Learning rate')
    parser.add_argument('--scheduler-type', type=str, default='warmup', choices=['warmup'], help='Scheduler type')
    parser.add_argument('--warmup-steps', type=int, default=4000, help='Warmup steps')
    parser.add_argument('--seed', type=int, default=42, help='Manual seed for torch')

    # text config
    parser.add_argument('--emsize', type=int, default=200, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=100,
                        help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')

    # combine model config
    parser.add_argument('--combine', type=str, default='concat', choices=['concat'],
                        help='Hidden dim size for the combine model')
    parser.add_argument('--d-model', type=int, default=512, help='Hidden dim size for the combine model')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='Dropout rate for the combine model')
    parser.add_argument('--board-embed-size', type=int, default=128, help='Size of board embedding. '
                                                                          'This depends on which katago model you use.')

    return parser


def evaluate(session, combine_model, board_model, text_model, criterion, val_dataloader, batch_size, device):
    combine_model.eval()
    batches = tqdm(enumerate(val_dataloader))
    total_correct = 0
    total_loss = 0
    total_total = 0
    for i_batch, sampled_batched in batches:
        color = sampled_batched[1]['color']
        board = sampled_batched[1]['board'].numpy()
        text = sampled_batched[1]['text']
        label = sampled_batched[1]['label'][:, None].to(device)

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
    accuracy = float(total_correct) / total_total
    loss_avg = float(total_loss) / total_total
    combine_model.train()
    return accuracy, loss_avg


def main():
    parser = parse_args()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.track:
        wandb.init(project=args.project_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    config = {'data_dir': args.data_dir,
              'device': device,
              }
    train_set = GoDataset(config, split='train')
    val_set = GoDataset(config, split='val')

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load pretrained katago model
    board_model, model_variables_prefix, model_config_json = katago.get_model(args.katago_model_dir)
    saver = tf.train.Saver(
        max_to_keep=10000,
        save_relative_paths=True,
    )

    # Load pretrained text Tranformer model
    ntokens = train_set.vocab_size # the size of vocabulary

    text_model = TransformerModel_extractFeature(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    if torch.cuda.is_available():
        text_model = text_model.cuda()
    for param in text_model.parameters():
        param.requires_grad = False  # freeze params in the pretrained model

    # Construct the model
    combine_model = PretrainedCombineModel(combine=args.combine,d_model=args.d_model, dropout_p=args.dropout_p,
                                           nhid=args.nhid, emsize=args.emsize, board_embed_size=args.board_embed_size)
    if torch.cuda.is_available():
        combine_model = combine_model.cuda()

    if args.track:
        wandb.watch(combine_model)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(combine_model.parameters(), args.learning_rate)
    if args.scheduler_type == 'warmup':
        lr_scheduler = LambdaLR(
            optimizer,
            WarmupLRSchedule(
                args.warmup_steps
            )
        )
    else:
        raise ValueError('Unknown scheduler type: ', args.scheduler_type)

    num_batches = len(train_set) / args.batch_size
    epochs = tqdm(range(args.num_epoch))
    loss_accumulate = 0
    count_accumulate = 0

    modules = {'combine_model': combine_model,
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler}

    val_loss_history = []

    # training
    combine_model.train()
    with tf.Session() as session:
        saver.restore(session, model_variables_prefix)
        step = 0
        for epoch in epochs:
            epochs.set_description(f'Epoch {epoch}')
            batches = tqdm(enumerate(train_dataloader))
            for i_batch, sampled_batched in batches:
                step += 1
                batches.set_description(f'batch {i_batch}/{num_batches}')
                color = sampled_batched[1]['color']
                board = sampled_batched[1]['board'].numpy()
                text = sampled_batched[1]['text']
                label = sampled_batched[1]['label'][:, None].to(device)

                try:
                    board_features = torch.tensor(
                        katago.extract_intermediate_optimized.extract_features_batch(session, board_model, board, color)).to(
                        device)
                except IllegalMoveError:
                    print(f"IllegalMoveError, skipped batch {sampled_batched[0]}")
                    continue

                text_features = torch.tensor(get_comment_features.extract_comment_features(text_model, text.to(device), args.batch_size, device)).to(device)

                logits = combine_model(board_features, text_features)
                loss = criterion(logits, label.type_as(logits))
                loss_accumulate += loss
                count_accumulate += sampled_batched[1]['label'].shape[0]

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if i_batch % args.checkpoint_interval == 0:
                    checkpoint_path = checkpoint(epoch, i_batch, modules, args.experiment_dir, max_checkpoints=args.max_checkpoints)

                    loss_avg = float(loss_accumulate) / count_accumulate
                    val_acc, val_loss = evaluate(session, combine_model, board_model, text_model, criterion, val_dataloader, args.batch_size, device)
                    val_loss_history.append(val_loss)
                    if val_loss >= max(val_loss_history):  # best
                        dirname = os.path.dirname(checkpoint_path)
                        basename = os.path.basename(checkpoint_path)
                        best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
                        shutil.copy2(checkpoint_path, best_checkpoint_path)
                    if args.track:
                        wandb.log({'Val Accuracy': val_acc,
                                   'Val Loss': val_loss,
                                   'Train Loss': loss_avg,
                                   'lr': lr_scheduler.get_lr(),
                                   'epoch': epoch,
                                   'step': step})
                    loss_accumulate = 0
                    count_accumulate = 0


if __name__ == '__main__':
    main()
