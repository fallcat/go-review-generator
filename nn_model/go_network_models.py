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

# set up the text for LSTM 
train_comment_df = pd.DataFrame()
train_comment_df['label'] = y
train_comment_df['comment'] = X_comment
first_n_words = 100
def trim_string(x):
    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x

#write to csv first and then load it to the iterator.. (don't want to write my own dataclass...)
train_comment_df['comment'] = train_comment_df['comment'].apply(trim_string)
train_comment_df['textlen'] = train_comment_df['comment'].str.len()

# string length == 0 cause trouble for later LSTM pack sequences
to_remove = train_comment_df.index[train_comment_df['textlen'] == 0].tolist()
for i in sorted(to_remove, reverse=True):
    del X_comment[i]
    del X_move[i]
    del X_board[i]
    del y[i]

# now create new train_comment copy with some obs deleted
train_comment_new_df = pd.DataFrame()
train_comment_new_df['label'] = y
train_comment_new_df['comment'] = X_comment
train_comment_new_df.to_csv('train_comment_df.csv', index = False)

# now set fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(lower=True, include_lengths=True, batch_first=True)
fields = [('label', label_field), ('comment', text_field)]

# now finally.. load dataset to iterators
train_comment_reader = TabularDataset(path='train_comment_df.csv', format='CSV', fields=fields, skip_header=True)

train_commen_loader = BucketIterator(train_comment_reader, batch_size=32, sort_key=lambda x: len(x.text), sort=False, sort_within_batch=False)
text_field.build_vocab(train_comment_reader, min_freq=10)

final_train = data_utils.TensorDataset(torch.Tensor(np.array(X_board)), torch.Tensor(np.array(X_move)), torch.tensor(np.array(y), dtype=torch.long))
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
        for i, data in enumerate(zip(cycle(train_loader), train_commen_loader), 0):
            # get the inputs
            data_1, data_2 = data

            x_board, x_move, labels = data_1
            (comment, commentlen) = data_2.comment

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(x_board, x_move, comment, commentlen)
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


class CNNLSTM(nn.Module):
    def __init__(self, dimension=64):
        super(CNNLSTM, self).__init__()        
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
        self.fc1 = nn.Linear(253, 32)
        self.classifier = nn.Linear(32, 2)

        # layer for comments
        self.embedding = nn.Embedding(len(text_field.vocab), 100)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=100, hidden_size=dimension, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2*dimension, 1)
        
    def forward(self, x_board, x_move, text, text_len):
        # reshape x_board, x_move
        x1 = x_board.unsqueeze(1)
        x2 = x_move.unsqueeze(1)
        x2 = x2.unsqueeze(3)

        x1 = self.features1(x1)
        x2 = self.features2(x2)

        text_emb = self.embedding(text)
        #print(text_len)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        x3 = self.fc(text_fea)
        #text_fea = torch.squeeze(text_fea, 1)
        #text_out = torch.sigmoid(text_fea)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.classifier(x)
        return x

net_cnnlstm = CNNLSTM()

# set up criteria
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_cnnlstm.parameters())

model, losses, accuracies = train_model(net_cnnlstm, criterion, optimizer, n_epochs = 20)