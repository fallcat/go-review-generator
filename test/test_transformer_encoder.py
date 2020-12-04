import sys
sys.path.append('..')
sys.path.append('.')

from transformer_encoder import get_comment_features

from transformer_encoder.model import *
from transformer_encoder import data_process
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
data_dir = "data_splits_final/"
# load data
# be careful. Since the whole dataset is too big, here I try to load them separately.
'''
train_comments,train_labels,vocab_size = prepare_comment(data_dir+"train.choices.pkl", data_dir+"train_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
train_comments = train_comments.to(device)
print(train_comments.shape)
'''
val_comments,val_labels,vocab_size = data_process.prepare_comment(data_dir+"val.choices.pkl", data_dir+"val_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
val_comments = val_comments.to(device)
'''
test_comments,test_labels,_ = prepare_comment(data_dir+"test.choices.pkl", data_dir+"test_comments.tok.32000.txt", data_dir+"vocab.32000", cutoff=5)
test_comments = test_comments.to(device)
print(test_comments.shape)
'''

batch_size = 64
ntokens = vocab_size # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 100 # the dimension of the feedforward network pretrained_combine in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel_extractFeature(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.0001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


features = get_comment_features.extract_comment_features(model, val_comments[:10], batch_size, device)

print(np.array(features).shape)