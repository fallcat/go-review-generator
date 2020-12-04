import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
from pretrained_combine import *
from transformer_encoder.data_process import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
data_dir = "../../data_splits_final/"

train_comments,train_labels,vocab_size = prepare_comment(data_dir+"train.choices.pkl", data_dir+"train_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
train_comments = train_comments.to(device)
print(train_comments.shape)

val_comments,val_labels,_ = prepare_comment(data_dir+"val.choices.pkl", data_dir+"val_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
val_comments = val_comments.to(device)


batch_size = 64 #batch size
ntokens = vocab_size # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 100 # the dimension of the feedforward network pretrained_combine in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
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
    output = model(data, src_mask)
    loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(pretrained_combine.parameters(), 0.5)
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
  eval_model.eval() # Turn on the evaluation mode
  total_loss = 0.
  #ntokens = len(TEXT.vocab.stoi)
  src_mask = model.generate_square_subsequent_mask(batch_size).to(device)
  with torch.no_grad():
    for i in range(0, data_source.size(0), batch_size):
      data = data_source[i:i+batch_size]
      data = torch.transpose(data,0,1)
      targets = data
      if data.size(0) != batch_size:
        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
      output = eval_model(data, src_mask)
      output_flat = output.view(-1, ntokens)
      total_loss += len(data) * criterion(output_flat, targets.reshape(-1)).item()
  return total_loss / (len(data_source) - 1)



best_val_loss = float("inf")
epochs = 5 # The number of epochs
best_model = None
save_path = "./model_weights"

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_comments)
    val_loss = evaluate(model, val_comments)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(best_model.state_dict(), save_path)
    #scheduler.step()

'''
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
'''
