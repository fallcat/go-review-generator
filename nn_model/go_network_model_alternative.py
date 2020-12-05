import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data_utils
import nltk

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

import time
import os
import tqdm

import pickle
import os
import time
import tqdm
import spacy
import en_core_web_sm
import pandas as pd
from itertools import cycle
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data_utils
import nltk
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# os.chdir('D:/GoReview')
dataset_dir = "data_splits_final"

file_obj = open(dataset_dir + '/train.pkl', 'rb')
train_board = pickle.load(file_obj)

file_obj = open(dataset_dir + '/train_comments.tok.32000.txt', 'r', encoding='utf-8')
train_comment = file_obj.readlines()

file_obj = open(dataset_dir + '/train.choices.pkl', 'rb')
train_choice = pickle.load(file_obj)

def init_vocab(vocab_file, cutoff):
  vocab = {}
  i = 2
  with open(vocab_file, encoding='utf8') as f:
    for line in f:
      line = line.strip().split(' ')
      token = line[0]
      count = int(line[1])
      if count > cutoff:
        vocab[token] = i
        i += 1
  return vocab

train_vocab = init_vocab(dataset_dir + '/vocab.32000', 0)

Max_Length = 100
comment_num = []
comment_length = []
for i in range(len(train_comment)):
    line = train_comment[i].strip().split(' ')
    convert = [train_vocab.get(token, 1) for token in line]
    if len(convert) > Max_Length:
        convert = convert[:Max_Length]
        c_len = len(convert)
    else:
        c_len = len(convert)
        convert += [0] * (Max_Length - c_len)
    if c_len==0:
        raise ValueError("Comment length is 0")
    comment_num.append(convert)
    comment_length.append(c_len)


def prepare_data(board_obj, comment_obj, comment_vec_obj, comment_len_obj, choice_obj):
    """
    initial process for the data
    """
    boards = board_obj['boards']
    X_move = []
    X_board = []
    X_comment = []
    X_comment_vec = []
    X_comment_len = []
    y = []

    players = {'b': 0, 'w':1}

    for i in range(len(choice_obj['answers'])):
        answer_index = choice_obj['answers'][i]
        choice_list = choice_obj['choice_indices'][i]
        pos_index = choice_list[answer_index]  # select positive from the choice list 

        b_pos = boards[pos_index]
        board_features = [b_pos[0], b_pos[1], players[b_pos[2]]]
        X_move.append(board_features)
        X_board.append(b_pos[3])
        X_comment.append(comment_obj[pos_index])
        X_comment_vec.append(comment_vec_obj[pos_index])
        X_comment_len.append(comment_len_obj[pos_index])
        y.append(1)

        # now for negative comment 
        if answer_index == 0:
            neg_idx = choice_list[1]
        else:
            neg_idx = choice_list[0]
        b_neg = boards[neg_idx]
        board_features_neg = [b_neg[0], b_neg[1], players[b_neg[2]]]
        X_move.append(board_features_neg)
        X_board.append(b_neg[3])
        X_comment.append(comment_obj[neg_idx])
        X_comment_vec.append(comment_vec_obj[neg_idx])
        X_comment_len.append(comment_len_obj[neg_idx])
        y.append(0)

    return X_board, X_move, X_comment, X_comment_vec, X_comment_len, y

X_board, X_move, X_comment, X_comment_vec, X_comment_len, y = prepare_data(train_board, train_comment, comment_num, comment_length, train_choice)


final_train = data_utils.TensorDataset(torch.Tensor(np.array(X_board)), \
     torch.Tensor(np.array(X_move)), \
     torch.tensor(np.array(X_comment_vec), dtype = torch.long),\
     torch.tensor(np.array(X_comment_len), dtype=torch.long),
     torch.tensor(np.array(y), dtype=torch.float32))
train_loader = data_utils.DataLoader(final_train, batch_size = 64, shuffle=True)


def train_model(net, criterion, optimizer, n_epochs):
    losses = []
    accuracies = []
    minibatch_accuracies = []

    corrected = 0
    total = 0
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times
        since = time.time()
        running_loss = 0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            running_loss_bat = 0

            x1, x2, x3, x4, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(x1, x2, x3, x4)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = (outputs >= 0.5).int()

            accuracy = (labels == pred).float().mean() # corrected / total

            # print statistics
            running_loss += loss.item()
            running_loss_bat += loss.item()
            running_correct += (labels==pred).sum().item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('Epoch: %d, Batch: %5d, loss: %.3f' %
                    (epoch + 1, i + 1, running_loss_bat / 200))
                print('Accuracy', accuracy.item())
                minibatch_accuracies.append(accuracy.item())
                running_loss_bat = 0.0

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 / 64 * running_correct / len(train_loader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
            
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

    print('Finished Training')
    return net, losses, accuracies, minibatch_accuracies

   
def model_eval(net):
    """ evaluate test set 
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return test_accuracy

vocab_size = len(train_vocab)+2

class MyModel(nn.Module):
    def __init__(self, dimension=64):
        super(MyModel, self).__init__()        
        # layers for board
        self.features1 = nn.Sequential(
            nn.Linear(19, 19*2),
            #nn.Conv2d(1, 19, 5, 1, 2),
            #nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Linear(19*2, 19*2),
        )
        # layers for move & player
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.ReLU())

        self.fc11 = nn.Linear(19, 19*19)
        self.fc12 = nn.Linear(19*19, 19)    
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.bil = nn.Bilinear(6862, 128, 64)

        # layer for comments
        self.embedding = nn.Embedding(vocab_size, 100)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=100, hidden_size=dimension, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)
        
    def forward(self, x_board, x_move, text, text_len):
        # reshape x_board, x_move
        #x1 = x_board.unsqueeze(1)
        #x2 = x_move.unsqueeze(1)
        #x2 = x2.unsqueeze(3)
        #x1 = self.features1(x1)
        #x2 = self.features2(x2)

        x1 = self.sp(self.fc11(x_board))
        #x1 = self.sp(self.fc12(x1))

        text_emb = self.embedding(text)
        #print(text_len)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        x3 = torch.cat((out_forward, out_reverse), 1)
        #x3 = self.drop(out_reduced)

        #x3 = self.fc(text_fea)
        #text_fea = torch.squeeze(text_fea, 1)
        #text_out = torch.sigmoid(text_fea)

        x1 = x1.view(x1.size(0), -1)
        x2 = x_move.view(x_move.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.bil(x, x3)
        #x = self.drop(x)
        x = self.drop(self.relu(self.fc1(x))) 
        x = self.drop(self.relu(self.fc2(x)))
  
        final_out = torch.sigmoid(self.fc3(x))
        return final_out
   

net_cnnlstm_test = MyModel()

# set up criteria
criterion = nn.BCELoss()
optimizer = optim.Adam(net_cnnlstm_test.parameters())

model, losses, accuracies, acc_mb = train_model(net_cnnlstm_test, criterion, optimizer, n_epochs = 5)

print(time.time)
#output = net_cnnlstm_test(x1,x2,x3,x4)
#pred = (outputs >= 0.5).int()
#accuracy = (labels == pred).float().mean() # corrected / total