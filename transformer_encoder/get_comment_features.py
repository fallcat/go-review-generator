import sys
sys.path.append('.')
sys.path.append('..')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
from pretrained_combine import *
from transformer_encoder.data_process import *
import os
import pickle


def load_model():
  '''
  batch_size = 64
  ntokens = vocab_size # the size of vocabulary
  emsize = 200 # embedding dimension
  nhid = 100 # the dimension of the feedforward network pretrained_combine in nn.TransformerEncoder
  nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead = 2 # the number of heads in the multiheadattention models
  dropout = 0.2 # the dropout value
  '''
  model = TransformerModel_extractFeature(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
  
  save_path = "./model_weights"
  if not torch.cuda.is_available():
    model.load_state_dict(torch.load(save_path,map_location=lambda storage, loc: storage))   #load to CPU
  else:
    model.load_state_dict(torch.load(save_path))    #load to GPU
  return model

def extract_comment_features(eval_model, data_source, batch_size, device):
  '''
  this function will return:
  whole_feature: (number_sample,sentence_length,nhid)
  '''
  eval_model.eval() # Turn on the evaluation mode
  src_mask = eval_model.generate_square_subsequent_mask(batch_size).to(device)
  whole_feature = []
  with torch.no_grad():
    for i in range(0, data_source.size(0), batch_size):
      data = data_source[i:i+batch_size]
      data = torch.transpose(data,0,1)
      targets = data
      if data.size(0) != batch_size:
        src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
      features,_ = eval_model(data, src_mask)
      features = torch.transpose(features,0,1)       # the one you need
      whole_feature.extend(features.tolist())        # convert to list
  return whole_feature



if __name__ == "__main__":
  #-----------------------------------Using Example-----------------------------------------------------------

  os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #device = torch.device("cpu")
  data_dir = "../../data_splits_final/"

  # load data
  # be careful. Since the whole dataset is too big, here I try to load them separately.
  val_comments,val_labels,vocab_size = prepare_comment(data_dir+"val.choices.pkl", data_dir+"val_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
  val_comments = val_comments.to(device)

  val_comments,val_labels,vocab_size = prepare_comment(data_dir+"val.choices.pkl", data_dir+"val_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
  val_comments = val_comments.to(device)



  batch_size = 64
  ntokens = vocab_size # the size of vocabulary
  emsize = 200 # embedding dimension
  nhid = 100 # the dimension of the feedforward network pretrained_combine in nn.TransformerEncoder
  nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
  nhead = 2 # the number of heads in the multiheadattention models
  dropout = 0.2 # the dropout value

  model = load_model()
  #Here we only load the first batch as an example
  example = val_comments[:batch_size]
  example_features = extract_comment_features(model, example, batch_size, device)

  print(np.asarray(example_features).shape)





