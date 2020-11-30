import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
from model import *
from transformer_encoder.data_process import *
import os
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
data_dir = "./data_splits_final/"


# load data
# be careful. Since the whole dataset is too big, here I try to load them separately. 
'''
train_comments,train_labels,vocab_size = prepare_comment(data_dir+"train.choices.pkl", data_dir+"train_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
train_comments = train_comments.to(device)
print(train_comments.shape)
'''
val_comments,val_labels,vocab_size = prepare_comment(data_dir+"val.choices.pkl", data_dir+"val_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
val_comments = val_comments.to(device)
'''
test_comments,test_labels,_ = prepare_comment(data_dir+"test.choices.pkl", data_dir+"test_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
test_comments = test_comments.to(device)
print(test_comments.shape)
'''

batch_size = 64
ntokens = vocab_size # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel_extractFeature(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.0001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(train_comments):
  model.train() # Turn on the train mode
  total_loss = 0.
  start_time = time.time()
  #ntokens = len(TEXT.vocab.stoi)
  src_mask = model.generate_square_subsequent_mask(batch_size).to(device)
  for batch, i in enumerate(range(0, train_comments.size(0), batch_size)):
    data = train_comments[i:i+batch_size]
    data = torch.transpose(data,0,1)
    targets = data
    optimizer.zero_grad()
    if data.size(0) != batch_size:
      src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
    features, output = model(data, src_mask)
    loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    log_interval = 200
    if batch % log_interval == 0 and batch > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | '
            'ms/batch {:5.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(
              epoch, batch, len(train_comments) // batch_size,
              elapsed * 1000 / log_interval,
              cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()

def evaluate(eval_model, data_source):
  '''
  this function will return:
  loss: scaler
  whole_feature: (number_sample,sentence_length,nhid)
  '''
  eval_model.eval() # Turn on the evaluation mode
  total_loss = 0.
  #ntokens = len(TEXT.vocab.stoi)
  src_mask = model.generate_square_subsequent_mask(batch_size).to(device)
  whole_feature = []
  with torch.no_grad():
    for i in range(0, data_source.size(0), batch_size):
      data = data_source[i:i+batch_size]
      data = torch.transpose(data,0,1)
      targets = data
      if data.size(0) != batch_size:
        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
      features, output = eval_model(data, src_mask)
      features = torch.transpose(features,0,1)       # the one you need
      whole_feature.extend(features.tolist())
      output_flat = output.view(-1, ntokens)
      total_loss += len(data) * criterion(output_flat, targets.reshape(-1)).item()
  return total_loss / (len(data_source) - 1), whole_feature



best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None
save_path = "transformer_encoder/model_weights"

#model.load_state_dict(torch.load(save_path).state_dict())
model.load_state_dict(torch.load(save_path,map_location=lambda storage, loc: storage))
#model.load_state_dict(torch.load(save_path))
save_batch = 2240

#---------------------------training set start-------------------------------------
'''
print("training set:")
os.system("mkdir train_batch_feature")

for count, i in enumerate(range(0, train_comments.size(0), save_batch)):
  train_loss, whole_feature = evaluate(model, train_comments[i:i+save_batch])
  temp_labels = train_labels[i:i+save_batch]
  d = {"features":[],"labels":[]}
  for j in range(0,len(whole_feature),2):
    d["features"].append([whole_feature[j],whole_feature[j+1]])
    d["labels"].append([temp_labels[j],temp_labels[j+1]])
  filename = "./train_batch_feature/batch_"+str(count)
  with open(filename, 'wb+') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
  if count % 20 == 0 and count > 0:
    print("current file:",filename)
#print(len(whole_feature),len(train_labels))

#---------------------------test set start-------------------------------------

print("test set:")
os.system("mkdir test_batch_feature")

for count, i in enumerate(range(0, test_comments.size(0), save_batch)):
  test_loss, whole_feature = evaluate(model, test_comments[i:i+save_batch])
  temp_labels = test_labels[i:i+save_batch]
  d = {"features":[],"labels":[]}
  for j in range(0,len(whole_feature),2):
    d["features"].append([whole_feature[j],whole_feature[j+1]])
    d["labels"].append([temp_labels[j],temp_labels[j+1]])
  filename = "./test_batch_feature/batch_"+str(count)
  with open(filename, 'wb+') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
  if count % 20 == 0 and count > 0:
    print("current file:",filename)
'''

#---------------------------validation set start-------------------------------------

print("val set:")

val_loss, whole_feature = evaluate(model, val_comments)
print(len(whole_feature),len(val_labels))

d = {"features":[],"labels":[]}
for i in range(0,len(whole_feature),2):
  d["features"].append([whole_feature[i],whole_feature[i+1]])
  d["labels"].append([val_labels[i],val_labels[i+1]])
with open('val_feature.pkl','wb+') as f:
  pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


