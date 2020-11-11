from os import listdir
from os.path import isfile, join
import random
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from itertools import islice
import nltk
nltk.download('stopwords') 
nltk.download('punkt')
##TODO: Line 71 and 98, take out debug for smaller data set

## argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help='directory where output files will be written to')
parser.add_argument('-dataDir', required=True, help='directory where data files will be read in from')
args = parser.parse_args()

## global variables
outdir, dataDir = args.outDir, args.dataDir
train_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'train' in f])
val_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'val' in f])
test_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'test' in f])
stones_dict = {'b':0,'w':1}

## sanity checks
print('\n---argparser---:')
for arg in vars(args):
    print(arg, getattr(args, arg), '\t', type(arg))

print('\n---train fnames---')
print(f'len: {len(train_fnames)}')
for f in train_fnames:
	print(f)

print('\n---val fnames---')
print(f'len: {len(val_fnames)}')
for f in val_fnames:
	print(f)

print('\n---test fnames---')
print(f'len: {len(test_fnames)}')
for f in test_fnames:
	print(f)

## data filenames
train_boards_fname, train_text_fname, train_choices_fname = train_fnames[2], train_fnames[5], train_fnames[1]
val_boards_fname, val_text_fname, val_choices_fname = val_fnames[2], val_fnames[5], val_fnames[1]
test_boards_fname, test_text_fname, test_choices_fname = test_fnames[2], test_fnames[5], test_fnames[1]


def create_board_feature_matrix(fname):

	## I might need to further split the transform data, not sure

	start = time.time()

	global stones_dict
	temp = []
	# dict_keys(['boards', 'steps'])
	X = DictVectorizer(sparse=False)
	file_obj = open(fname, "rb")
	data = pkl.load(file_obj)['boards']
	data_iter = iter(data)
	for board in data_iter:
		flattened_board = board[3].flatten().tolist()
		modified_board = [board[0],board[1],stones_dict[board[2]]]+flattened_board
		temp.append(dict(zip(range(len(modified_board)),modified_board)))
	X.fit(temp)

	feature_time = time.time() - start
	print('\nTime elapsed making board feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(feature_time))))

	# print(f'\nX.fit_transform(temp): \n{X.fit_transform(temp)}')
	# print(f'\nX.transform(temp): \n{X.transform(temp)}')
	return X.transform(temp)


def create_text_feature_matrix(fname):

	start = time.time()

	matrix = CountVectorizer(max_features=1000)

	file_obj = open(fname,"r+")
	data = file_obj.readlines()
	data_iter = iter(data)
	sentences = [sent.strip() for sent in data_iter]
	X = matrix.fit_transform(sentences).toarray() ## transforms sentences to unigram features

	text_time = time.time() - start
	print('\nTime elapsed making text feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(text_time))))

	return X

def get_examples(boards_mat, text_mat, choices_fname):

	start = time.time()
	print(f'\nboards_mat.shape: {boards_mat.shape} text_mat.shape: {text_mat.shape}')

	X, y = [],[]
	file_obj = open(choices_fname, "r+")
	choices = file_obj.read().splitlines()
	choices_iter = iter(choices)
	for line in choices_iter:
		cur_indices = [int(idx) for idx in line.split()]
		pos_idx = cur_indices[cur_indices[-1]]
		## positive examples
		X.append((boards_mat[pos_idx],text_mat[pos_idx]))
		y.append(1)
		## negative examples
		cur_indices.remove(pos_idx)
		neg_iter = iter(cur_indices[:-1])
		for neg_idx in neg_iter:
			X.append((boards_mat[neg_idx],text_mat[neg_idx]))
			y.append(0)

	examples_time = time.time() - start
	print(f'\n(len(X), len(y)): {len(X), len(y)}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return np.array(X), np.array(y)

def main(X, y, X_test, y_test):

	# Logistic Regression 
	clf = LogisticRegression(random_state=0).fit(X, y)

	# Predict Class
	y_pred = clf.predict(X_test)

	# Accuracy 
	accuracy = accuracy_score(y_pred, y_test)

	return accuracy

################
### HELPER FUNCS
################

def progress_in_batches(name, iterator, total_length, increment):
    count = 0
    while True:
        next_batch = list(islice(iterator, increment))
        if len(next_batch) > 0:
            count += len(next_batch)
            print(f'\nGenerating features for {name}... percentage processed {(count/total_length)*100}')
            print(f'next_batch:{next_batch}')
            yield next_batch
        else:
            break



if __name__ == '__main__':

	start = time.time()

	## TRAIN
	# get feature matrices
	train_board_feature_matrix = create_board_feature_matrix(train_boards_fname) ## 'numpy.ndarray', <class 'sklearn.feature_extraction._dict_vectorizer.DictVectorizer'>
	train_text_feature_matrix  = create_text_feature_matrix(train_text_fname) ## <class 'numpy.ndarray'>
	print(f'\ntrain_board_feature_matrix.shape: {train_board_feature_matrix.shape}\ntrain_text_feature_matrix.shape: {train_text_feature_matrix.shape}')
	# get training examples
	X_train, y_train = get_examples(train_board_feature_matrix, train_text_feature_matrix, train_choices_fname)
	X_train_shuffled, y_train_shuffled = shuffle(X_train,y_train)

	## VAL
	# get feature matrices
	val_board_feature_matrix = create_board_feature_matrix(val_boards_fname)
	val_text_feature_matrix  = create_text_feature_matrix(val_text_fname)
	# get validation examples
	X_val,  y_val = get_examples(val_board_feature_matrix, val_text_feature_matrix, val_choices_fname)

	## TEST
	# get feature matrices
	# test_board_feature_matrix = create_board_feature_matrix(text_boards_fname)
	# test_text_feature_matrix  = create_text_feature_matrix(text_text_fname)
	# get test examples

	train_acc = main(X_train_shuffled, y_train_shuffled, X_train, y_train)
	val_acc = main(X_train_shuffled, y_train_shuffled, X_val, y_val)

	main_time = time.time() - start

	print('\n---baseline stats---')
	print(f'train accuracy:\t{train_acc}')
	print(f'val accuracy:\t{val_acc}')
	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))



