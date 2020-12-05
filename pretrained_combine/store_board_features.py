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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_epoch = 10

# Dataloader
config = {'data_dir': data_dir,
          'katago_model_dir': katago_model_dir,
          'device': device}
train_set = GoDataset(config, split='train')
val_set = GoDataset(config, split='val')
test_set = GoDataset(config, split='test')

