import sys
sys.path.append('.')
sys.path.append('..')
from os import listdir
from os.path import isfile, join
import random
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse
import time
import random
from itertools import islice, product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, fbeta_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
import katago
from transformer_encoder import get_comment_features
from transformer_encoder.model import *
from transformer_encoder.data_process import *
from transformer_encoder import data_process
import os
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'line 32 device: {device}')
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords') 
nltk.download('punkt')
## TODO: Try glove or pymagnitude word embedding, sum the average of the word embeddings
## TODO: Try pretrained text embeddings from transformer encoder
## TODO: CTRL-F exit()

## argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-notes', required=True, help='run id notes to be printed out at the top of the argparser to help distinguish from other runs.')
parser.add_argument('-outDir', required=True, help='directory where output files will be written to')
parser.add_argument('-dataDir', required=True, help='directory where data files will be read in from')
args = parser.parse_args()

## global variables
outdir, dataDir = args.outDir, args.dataDir
train_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'train' in f])
val_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'val' in f])
test_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'test' in f])
vocab_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'vocab' in f])

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

print('\n---vocab fnames---')
print(f'len: {len(vocab_fnames)}')
for f in vocab_fnames:
	print(f)
print('\n')

## data filenames
train_boards_fname, train_text_fname, train_choices_fname, train_choices_pkl_fname = train_fnames[2], train_fnames[3], train_fnames[1], train_fnames[0]
val_boards_fname, val_text_fname, val_choices_fname, val_choices_pkl_fname = val_fnames[2], val_fnames[3], val_fnames[1], val_fnames[0]
test_boards_fname, test_text_fname, test_choices_fname, test_choices_pkl_fname = test_fnames[2], test_fnames[3], test_fnames[1], test_fnames[0]
vocab_fname = vocab_fnames[0]


def create_pretrained_board_feature_matrix(session, model, fname):

	start = time.time()
	# dict_keys(['boards', 'steps'])
	X = DictVectorizer(dtype=np.float32, sparse=False)
	## generators for faster processing, lazy loading for memory efficiency
	board_data, color_data = next(lazy_pickle_load(fname)), next(lazy_pickle_load(fname))  
	batched_boards, batched_colors = np.array([x[3] for x in board_data]), np.array([x[2] for x in color_data])
	katago_boards = katago.extract_intermediate_optimized.extract_features_batch(session, model, batched_boards, batched_colors) ## katago_board['trunk'] is <class 'numpy.ndarray'> shape (19,19,256)
	# board_features = torch.tensor(katago.extract_features_batch(session, board_model, board, color)).to(device)
	# print(f'katago_boards.shape: {katago_boards.shape}') ## katago_boards.shape: (20676, 19, 19, 128)
	flattened_katago_boards = np.mean(katago_boards,axis=-1).reshape(katago_boards.shape[0], katago_boards.shape[1]*katago_boards.shape[2]) 
	# print(f'flattened_katago_boards.shape: {flattened_katago_boards.shape}') ##flattened_katago_boards.shape: (20676, 361)
	temp = [dict(zip(range(len(board)),board)) for board in flattened_katago_boards]

	X.fit(temp)
	X_nparr = X.transform(temp)
	X_tensor = torch.from_numpy(X_nparr) #.type(torch.cuda.FloatTensor)
	# X_tensor = torch.Tensor(X.transform(temp), dtype=torch.float32)
	print(f'BEFORE X_tensor.is_cuda: ', X_tensor.is_cuda)
	print(f'line 108 device: {device}')
	X_tensor = X_tensor.to(device)
	print(f'AFTER X_tensor.is_cuda: ', X_tensor.is_cuda)

	feature_time = time.time() - start
	# print(f'type(board_feature_matrix): {type(X_nparr)}') ##<class 'numpy.ndarray'> (20676, 361)
	# print(f'board_feature_matrix.shape: {X_nparr.shape}')
	print(f'type(tensor board_feature_matrix): {type(X_tensor)}') ##<class 'torch.Tensor'> torch.Size([20676, 361])
	print(f'tensor board_feature_matrix.shape: {X_tensor.shape}')
	print('Time elapsed making board feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(feature_time))))

	return X_tensor


def create_pretrained_text_feature_matrix(choices_pkl_fname, text_fname, vocab_fname, cutoff=5):

	start = time.time()

	comments, labels, vocab_size = prepare_comment(choices_pkl_fname, text_fname, vocab_fname, cutoff)
	comments, labels = comments.to(device), np.array(labels) ##<class 'torch.Tensor'>, <class 'numpy.ndarray'>
	true_comments, neg_comments = comments[np.where(labels==1)], comments[np.where(labels==0)]
	print('type(true_comments), type(neg_comments): ', type(true_comments), type(neg_comments))
	print('type(true_comments), type(neg_comments): ', type(true_comments[0]), type(neg_comments[0]	))
	true_comments.to(device)
	neg_comments.to(device)

	# matrix = CountVectorizer(dtype=np.float32, max_features=1000)
	# data = next(lazy_load(text_fname))
	# sentences = [sent.strip() for sent in data]
	# X = matrix.fit_transform(sentences).toarray() ## transforms sentences to unigram features

	text_time = time.time() - start
	print(f'true_comments.shape: {true_comments.shape}\tneg_comments.shape: {neg_comments.shape}')
	print('Time elapsed making text feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(text_time))))

	return true_comments, neg_comments #<class 'torch.Tensor'> torch.Size([20676, 100])

def get_examples(boards_matrix, pos_text_matrix, neg_text_matrix, fnames):

	start = time.time()

	X, y = [],[]

	for i, (board_feat_arr, pos_text_feat_arr, neg_text_feat_arr, fname) in enumerate(zip(boards_matrix, pos_text_matrix, neg_text_matrix, fnames)):
		print(f'board_feat_arr.shape: {board_feat_arr.shape}')
		print(f'len(pos_text_feat_arr): {len(pos_text_feat_arr)}\tlen(neg_text_feat_arr): {len(neg_text_feat_arr)}')
		print(f'cur fname: {fname}\n')
		choices = next(lazy_load(fname))
		for line in choices:
			indices_str, pos_idx = line.split('\t')
			# cur_indices = [int(idx) for idx in indices_str.split()]
			# del cur_indices[int(pos_idx)]
			## append positive example
			a = board_feat_arr[int(pos_idx)]
			b = pos_text_feat_arr[i]
			print(type(a))
			print(a.shape)
			print(type(b))
			print(b.shape)
			print(f'a.is_cuda: {a.is_cuda}')
			print(f'b.is_cuda: {b.is_cuda}')
			X.append(torch.cat((board_feat_arr[int(pos_idx)],pos_text_feat_arr[i]))) ##pos_text_feat_arr[int(pos_idx)]
			y.append(1)
			## append negative example, select the 1st negative example from the group
			X.append(torch.cat((board_feat_arr[int(pos_idx)],neg_text_feat_arr[i])))
			y.append(0)
			break
		break
	exit()

	examples_time = time.time() - start
	X_nparr, y_nparr = X.numpy(), y.numpy()
	print(f'tensor X num_examples: {X.shape}\ttensor y num_examples: {y.shape}')
	print(f'X num_examples: {X_nparr.shape}\ty num_examples: {y_nparr.shape}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return X, y ##X_nparr, y_nparr - do I need to convert this to array or leave in tensor?

def automatic_grid_search(X, y, X_test, y_test, test_set=False):

	print(f'starting auto parameter search..')
	start = time.time()
	clf = None

	'''parameters settings explored
		1. parameters = [{'penalty': ['l1'], 'solver':['liblinear'], 'C': [0.001,0.01,0.1,1,10]}, \
						 {'penalty': ['l2'], 'solver':['lbfgs','liblinear','sag'], 'C': [0.001,0.01,0.1,1,10]}, \
						 {'penalty': ['elasticnet'], 'l1_ratio':[0.25, 0.5, 0.6],'solver':['saga'], 'C': [0.001,0.01,0.1,1,10]}]

		2. parameters = [{'penalty': ['l1'], 'solver':['liblinear'], 'C': [0.001,0.01,0.1,1,10]}]
		3. parameters = [{'penalty': ['l2'], 'solver':['lbfgs','liblinear','sag'], 'C': [0.001,0.01,0.1,1,10]}] 
		4. parameters = [{'penalty': ['elasticnet'], 'l1_ratio':[0.25, 0.5, 0.6],'solver':['saga'], 'C': [0.001,0.01,0.1,1,10]}]
		5. parameters = {'C': [0.001,0.01,0.1]}
	'''

	if test_set:

		scores = ['precision', 'recall', 'f1','accuracy']

		for score in scores:
			print("\n# Tuning hyper-parameters for %s" % score)
			print('\n')

			clf = GridSearchCV(LogisticRegression(random_state=0, max_iter=10000), parameters, scoring='%s_macro' % score)
			clf.fit(X, y)

			print("Best parameters set found on development set:")
			print('\n')
			print(clf.best_params_)
			print('\n')
			print("Grid scores on development set:")
			print('\n')
			means = clf.cv_results_['mean_test_score']
			stds = clf.cv_results_['std_test_score']
			for mean, std, params in zip(means, stds, clf.cv_results_['params']):
				print("%0.3f (+/-%0.03f) for %r"
					  % (mean, std * 2, params))
			print('\n')

			print("Detailed classification report:")
			print('\n')
			print("The model is trained on the full development set.")
			print("The scores are computed on the full evaluation set.")
			print('\n')
			y_true, y_pred = y_test, clf.predict(X_test)
			print(classification_report(y_true, y_pred))
			print('\n')

	else:

		clf = GridSearchCV(LogisticRegression(random_state=0, max_iter=10000), parameters)
		clf.fit(X, y)


	parameter_search_time = time.time() - start
	print(f'end auto parameter search..')
	print('Time elapsed in doing an automatic parameter search:\t%s\n' % (time.strftime("%H:%M:%S", time.gmtime(parameter_search_time))))

	return clf.score(X_test,y_test), clf.best_params_, clf.best_estimator_, clf.best_score_

def main(X, y, X_test, y_test, best_params):


################
### HELPER FUNCS
################

def progress_in_batches(name, iterator, total_length, increment):
	count = 0
	while True:
		next_batch = list(islice(iterator, increment))
		if len(next_batch) > 0:
			count += len(next_batch)
			print('\nCreating SQLiteDict {}... percentage processed {}'.format(name, (count/total_length)*100))
			yield next_batch
		else:
			break

def lazy_pickle_load(fname):
	file_obj = open(fname, "rb")
	contents = pkl.load(file_obj)['boards']
	file_obj.close()
	yield contents

def lazy_load(fname):
	file_obj = open(fname,"r+")
	contents = file_obj.readlines()
	file_obj.close()
	yield contents


if __name__ == '__main__':

	start = time.time()

	saved_model_dir = "katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/"
	model, model_variables_prefix, model_config_json = katago.get_model(saved_model_dir)

	saver = tf.train.Saver(
		max_to_keep=10000,
		save_relative_paths=True,
	)

	tf.compat.v1.disable_eager_execution()

	with tf.Session() as session:
		saver.restore(session, model_variables_prefix)

		# print('\n---TRAIN EXAMPLES---')
		# train_board_feature_matrix = create_pretrained_board_feature_matrix(session, model, train_boards_fname) ## 'numpy.ndarray', <class 'sklearn.feature_extraction._dict_vectorizer.DictVectorizer'>
		# pos_train_text_feature_matrix, neg_train_text_feature_matrix  = create_pretrained_text_feature_matrix(train_choices_pkl_fname, train_text_fname, vocab_fname) ## <class 'numpy.ndarray'>
		# X_train, y_train = get_examples([train_board_feature_matrix], [pos_train_text_feature_matrix], [neg_train_text_feature_matrix], [train_choices_fname]) ##<class 'numpy.ndarray'>, <class 'numpy.ndarray'>
		# file_obja, file_objb = open('data_pretrained_tensors/X_train.pkl', 'wb'), open('data_pretrained_tensors/y_train.pkl', 'wb')
		# pkl.dump(X_train, file_obja)
		# pkl.dump(y_train, file_objb)


		print('\n---VALIDATION EXAMPLES---')
		val_board_feature_matrix = create_pretrained_board_feature_matrix(session, model, val_boards_fname)
		pos_val_text_feature_matrix, neg_val_text_feature_matrix  = create_pretrained_text_feature_matrix(val_choices_pkl_fname, val_text_fname, vocab_fname)
		X_val,  y_val = get_examples([val_board_feature_matrix], [pos_val_text_feature_matrix], [neg_val_text_feature_matrix], [val_choices_fname])
		file_objc, file_objd = open('data_pretrained_tensors/X_val.pkl', 'wb'), open('data_pretrained_tensors/y_val.pkl', 'wb')
		pkl.dump(X_val, file_objc)
		pkl.dump(y_val, file_objd)
		exit()


		print('\n---TEST EXAMPLES---')
		test_board_feature_matrix = create_pretrained_board_feature_matrix(session, model, test_boards_fname)
		pos_test_text_feature_matrix, neg_test_text_feature_matrix  = create_pretrained_text_feature_matrix(test_choices_pkl_fname, test_text_fname, vocab_fname)
		X_test,  y_test = get_examples([test_board_feature_matrix], [pos_test_text_feature_matrix], [neg_test_text_feature_matrix], [test_choices_fname])
		file_obje, file_objf = open('data_pretrained_tensors/X_test.pkl', 'wb'), open('data_pretrained_tensors/y_test.pkl', 'wb')
		pkl.dump(X_test, file_obje)
		pkl.dump(y_test, file_objf)


		print('\n---TRAIN+VALIDATION COMBINED EXAMPLES FOR AUTOGRID SEARCH---')
		X_comb, y_comb = get_examples([train_board_feature_matrix,val_board_feature_matrix], [pos_train_text_feature_matrix,pos_val_text_feature_matrix], [neg_train_text_feature_matrix,neg_val_text_feature_matrix], [train_choices_fname,val_choices_fname])
		file_objg, file_objh = open('data_pretrained_tensors/X_comb.pkl', 'wb'), open('data_pretrained_tensors/y_comb.pkl', 'wb')
		pkl.dump(X_comb, file_objg)
		pkl.dump(y_comb, file_objh)

		print('\n---Logistic Regression Auto Grid In Progress---')
		# train_accuracy, train_best_params, train_best_estimator, train_best_score  = automatic_grid_search(X_train[:2000], y_train[:2000], X_train, y_train)
		# val_accuracy, val_best_params, val_best_estimator, val_best_score  = automatic_grid_search(X_train[:2000], y_train[:2000], X_val, y_val)
		best_params = {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
		test_accuracy, test_best_params, test_best_estimator, test_best_score  = automatic_grid_search(X_comb[:100], y_comb[:100], X_test, y_test, True)
		print(f'test_accuracy: {test_accuracy}\ttest_best_score: {test_best_score}')

		## display results
		print('\n---Results---')
		for title, params, acc in zip(['*test*'], [test_best_params],[test_accuracy]):
			print(f'{title}\tbest_model: {params}\taccuracy: {acc}')
	
	main_time = time.time() - start
	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))




