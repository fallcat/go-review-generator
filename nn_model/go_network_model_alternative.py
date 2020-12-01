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

# simple nlp for now
nlp_sm = en_core_web_sm.load()
train_comment_vec = []
for i in range(len(train_comment)):
    train_comment_vec.append(nlp_sm(train_comment[i]).vector)

def prepare_data(board_obj, comment_obj, comment_vec_obj, choice_obj):
    """
    initial process for the data
    """
    boards = board_obj['boards']
    X_move = []
    X_board = []
    X_comment = []
    X_comment_vec = []
    y = []

    players = {'b': 0, 'w':1}

    for i in range(len(choice_obj['answers'])):
        pos_idx = choice_obj['answers'][i]
        choice_list = choice_obj['choice_indices'][i]
        b = boards[i]
        board_features = [b[0], b[1], players[b[2]]]
        X_move.append(board_features)
        X_board.append(b[3])
        X_comment.append(comment_obj[choice_list[pos_idx]])
        X_comment_vec.append(comment_vec_obj[choice_list[pos_idx]])
        y.append(1)

        # now for negative comment 
        if pos_idx == 0:
            neg_idx = 1
        else:
            neg_idx = 0
        X_move.append(board_features)
        X_board.append(b[3])
        X_comment.append(comment_obj[choice_list[neg_idx]])
        X_comment_vec.append(comment_vec_obj[choice_list[neg_idx]])
        y.append(0)

    return X_board, X_move, X_comment, X_comment_vec, y

X_board, X_move, X_comment, X_comment_vec, y = prepare_data(train_board, train_comment, train_comment_vec, train_choice)


final_train = data_utils.TensorDataset(torch.Tensor(np.array(X_board)), torch.Tensor(np.array(X_move)), torch.tensor(np.array(X_comment_vec), dtype = torch.long), torch.tensor(np.array(y), dtype=torch.long))
train_loader = data_utils.DataLoader(final_train, batch_size = 32, shuffle=True)


def train_model(net, criterion, optimizer, n_epochs):
    losses = []
    accuracies = []

    corrected = 0
    total = 0
    for epoch in tqdm.tqdm(range(n_epochs)):  # loop over the dataset multiple times
        since = time.time()
        running_loss = 0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs

            x1, x2, x3, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(x1, x2, x3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs.data, 1)

            accuracy = (labels == pred).float().mean() # corrected / total

            # print statistics
            running_loss += loss.item()
            running_correct += (labels==pred).sum().item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('Epoch: %d, Batch: %5d, loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                print('Accuracy', accuracy.item())
                running_loss = 0.0

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 / 32 * running_correct / len(train_loader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
            
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

    print('Finished Training')
    return net, losses, accuracies

   
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


class MyModel(nn.Module):
    def __init__(self,  hidden_dim = 64):
        super(MyModel, self).__init__()        
        # layers for board
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),)
        # layers for move & player
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.ReLU(),)


        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(284, 32)
        self.classifier = nn.Linear(32, 2)

        self.embeddings = nn.Embedding(0, 100)
        self.lstm = nn.LSTM(100, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, x_board, x_move, text):
        # reshape x_board, x_move
        x1 = x_board.unsqueeze(1)
        x2 = x_move.unsqueeze(1)
        x2 = x2.unsqueeze(3)

        x1 = self.features1(x1)
        x2 = self.features2(x2)

        x3 = self.embeddings(text)
        x3 = self.dropout(x3)
        lstm_out, (ht, ct) = self.lstm(x3)
        x3 = self.linear(ht[-1])

        #text_fea = torch.squeeze(text_fea, 1)
        #text_out = torch.sigmoid(text_fea)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x3 = x3.reshape(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)
        return x

net_cnnlstm_test = MyModel()

# set up criteria
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_cnnlstm_test.parameters())

model, losses, accuracies = train_model(net_cnnlstm_test, criterion, optimizer, n_epochs = 20)

