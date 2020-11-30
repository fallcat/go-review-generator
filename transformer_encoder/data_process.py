import pickle
import torch
import numpy as np

MAX_LENGTH = 100
PAD_ID = 0
UNK_ID = 1

def init_vocab(vocab_file,cutoff):
  vocab = {}
  i = 2
  with open(vocab_file) as f:
    for line in f:
      line = line.strip().split(' ')
      token = line[0]
      count = int(line[1])
      if count > cutoff:
        vocab[token] = i
        i += 1
  return vocab

def sentence_to_id(sentence,vocab):
  return [vocab.get(token,UNK_ID) for token in sentence]

def read_comment_subword(comment_token_file,vocab_file,cutoff):
  comment_id = []
  vocab = init_vocab(vocab_file,cutoff)
  with open(comment_token_file) as f:
    for line in f:
      line = line.strip().split(' ')
      sentence = sentence_to_id(line,vocab)
      length = len(sentence)
      if length < MAX_LENGTH:
        sentence += [PAD_ID] * (MAX_LENGTH - length)
      elif length > MAX_LENGTH:
        sentence = sentence[:MAX_LENGTH]
        #raise ValueError("Comment length is larger than {}\n".format(MAX_LENGTH))
      comment_id.append(sentence)
  return comment_id, len(vocab)+2

def oversample(sample,label,times):
  return sample * times, label * times

def prepare_comment(data_file, comment_token_file, vocab_file, cutoff=0):
  comment_id, vocab_size = read_comment_subword(comment_token_file,vocab_file,cutoff)
  comments = []
  labels = []
  with open(data_file,'rb') as input_file:
    data = pickle.load(input_file)
    for i in range(len(data['answers'])):
      #temp = []
      comment_indices = data['choice_indices'][i]
      label_index = data['answers'][i]
      #comment_indices = line[0].split(' ')
      #print(line)
      #label_index = comment_indices[int(line[-1])]
      '''
      for i in comment_indices:
        i = int(i)
        if i != label_index:
          comment = comment_id[i]
          label = 0
        else:
          comment = comment_id[i]
          label = 1
        #temp.append([comment,label])
      '''
      if label_index == 0:
        labels.extend([1,0])
        comments.extend([comment_id[comment_indices[label_index]],comment_id[comment_indices[1]]])
      else:
        labels.extend([0,1])
        comments.extend([comment_id[comment_indices[0]],comment_id[comment_indices[label_index]]])
  return torch.tensor(np.asarray(comments)),labels,vocab_size

def prepare_board(data_file):
  with open(data_file,'rb') as f:
    data = pickle.load(f)['boards']
  return data
'''
def read_data(board_file,comment_file, comment_token_file, vocab_file, cutoff=0):
  boards = prepare_board(board_file)
  comment_label = prepare_comment(comment_file, comment_token_file, vocab_file, cutoff)
'''  
