import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data_utils
import nltk


import numpy as np
import pickle

import time
import os
import tqdm

import pickle
import os
import time
import tqdm
#import spacy
#import en_core_web_sm
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
import torch.nn.functional as F

# os.chdir('D:/GoReview')
dataset_dir = "data_splits_final"

def raw_file_loader(dataset_dir, fboard, fcomment, fchoice):
    file_obj = open(dataset_dir + fboard, 'rb')
    raw_board = pickle.load(file_obj)

    file_obj = open(dataset_dir + fcomment, 'r', encoding='utf-8')
    raw_comment = file_obj.readlines()

    file_obj = open(dataset_dir + fchoice, 'rb')
    raw_choice = pickle.load(file_obj)

    return raw_board, raw_comment, raw_choice

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

def create_comment_vec(raw_comment, vocabu):
    Max_Length = 100
    c_num = []
    c_length = []
    for i in range(len(raw_comment)):
        line = raw_comment[i].strip().split(' ')
        convert = [vocabu.get(token, 1) for token in line]
        if len(convert) > Max_Length:
            convert = convert[:Max_Length]
            c_len = len(convert)
        else:
            c_len = len(convert)
            convert += [0] * (Max_Length - c_len)
        if c_len==0:
            raise ValueError("Comment length is 0")
        c_num.append(convert)
        c_length.append(c_len)
    return c_num, c_length


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


'''Load training set'''
train_board, train_comment, train_choice = raw_file_loader(dataset_dir, '/train.pkl', '/train_comments.tok.32000.txt', '/train.choices.pkl')
comment_num, comment_length = create_comment_vec(train_comment, train_vocab)
X_board, X_move, X_comment, X_comment_vec, X_comment_len, y = prepare_data(train_board, train_comment, comment_num, comment_length, train_choice)

final_train = data_utils.TensorDataset(torch.Tensor(np.array(X_board[0:100000])), \
     torch.Tensor(np.array(X_move[0:100000])), \
     torch.tensor(np.array(X_comment_vec[0:100000]), dtype = torch.long),\
     torch.tensor(np.array(X_comment_len[0:100000]), dtype=torch.long),
     torch.tensor(np.array(y[0:100000]), dtype=torch.float32))
train_loader = data_utils.DataLoader(final_train, batch_size = 64, shuffle=True)

test_board, test_comment, test_choice = raw_file_loader(dataset_dir, '/test.pkl', '/test_comments.tok.32000.txt', '/test.choices.pkl')
comment_num_test, comment_length_test = create_comment_vec(test_comment, train_vocab)
X_board_test, X_move_test, X_comment_test, X_comment_vec_test, X_comment_len_test, y_test = prepare_data(test_board, test_comment, comment_num_test, comment_length_test, test_choice)

final_test = data_utils.TensorDataset(torch.Tensor(np.array(X_board_test)), \
     torch.Tensor(np.array(X_move_test)), \
     torch.tensor(np.array(X_comment_vec_test), dtype = torch.long),\
     torch.tensor(np.array(X_comment_len_test), dtype=torch.long),
     torch.tensor(np.array(y_test), dtype=torch.float32))
test_loader = data_utils.DataLoader(final_test, batch_size = 64, shuffle=True)

def model_eval(net):
    """ evaluate test set 
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            var1,var2,var3,var4, labels = data
            outputs = net(var1,var2,var3,var4)
            labels = labels.unsqueeze(1)
            pred = (outputs >= 0.5).int()
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    test_accuracy = 100 * correct / total
    print('Accuracy of the network on the test sets: %.6f' % (100 * correct / total), flush = True)
    return test_accuracy


def train_model(net, criterion, optimizer, n_epochs):
    losses = []
    accuracies = []
    test_accuracies = []

    net.train()
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times
        since = time.time()
        running_loss = 0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):

            x1, x2, x3, x4, labels = data

            # zero the parameter gradients
            net.zero_grad()
            # forward + backward + optimize
            outputs = net(x1, x2, x3, x4)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = (outputs > 0.5).int()
            accuracy = (labels == pred).float().mean() # corrected / total

            # print statistics
            running_loss += loss.item()
            running_correct += (labels==pred).sum().item()

            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' %(epoch + 1, i + 1, running_loss / 200), flush = True)
                print('Accuracy', accuracy.item(), flush = True)
                losses.append(running_loss)
                accuracies.append(accuracy.item())
                running_loss = 0.0

        # evaluate models
        net.eval()
        test_acc = model_eval(net)
        test_accuracies.append(test_acc)

        # reset to train
        net.train()
        #scheduler.step(test_acc)

    print('Finished Training')
    return net, losses, accuracies, test_accuracies

   

vocab_size = len(train_vocab)+2


class MyModel(nn.Module):
    def __init__(self, dimension=64):
        super(MyModel, self).__init__()        
        # layers for board
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fcboard_1 = nn.Linear(in_features=32*19*19, out_features=19*19)

        self.fc11 = nn.Linear(19*19, 19*19)
        #self.fc12 = nn.Linear(19*19, 19)    
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.fc1 = nn.Linear(492, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.bil = nn.Bilinear(364, 128, 64)

        # layer for comments
        self.embedding = nn.Embedding(vocab_size, 100)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=100, hidden_size=dimension, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)
        
    def forward(self, x_board, x_move, text, text_len):
        # reshape x_board, x_move
        x1 = x_board.unsqueeze(1)
        
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(-1, 32*19*19)
        x1 = self.fcboard_1(x1)        

        text_emb = self.embedding(text)
        lstm_out, (h_n, c_n) = self.lstm(text_emb)
        x3 = torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)
        #x3 = self.drop(out_reduced)


        x1 = x1.view(x1.size(0), -1)
        x2 = x_move.view(x_move.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        #x = self.bil(x, x3)
        #x = self.drop(x)
        x = self.drop(F.relu(self.fc1(x))) 
        x = self.drop(F.relu(self.fc2(x)))
  
        final_out = torch.sigmoid(self.fc3(x))
        return final_out


net_cnnlstm_test = MyModel()

# set up criteria
criterion = nn.BCELoss()
optimizer = optim.Adam(net_cnnlstm_test.parameters(), lr=0.00001)

begin = time.time()
model, losses, accuracies, t_accuracies = train_model(net_cnnlstm_test, criterion, optimizer, n_epochs = 25)
total = time.time() - begin

#print(time.time)
print(losses, flush = True)
print(accuracies, flush = True)
print(t_accuracies, flush = True)

#output = net_cnnlstm_test(x1,x2,x3,x4)
#pred = (outputs >= 0.5).int()
#accuracy = (labels == pred).float().mean() # corrected / total