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
from sklearn.model_selection import GridSearchCV ##internal cross-validation
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, fbeta_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import katago
from transformer_encoder.get_comment_features import *
from transformer_encoder.model import *
from transformer_encoder.data_process import *
from transformer_encoder import data_process
import os
import torch
import h5py
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords') 
nltk.download('punkt')
## TODO: Try glove or pymagnitude word embedding, sum the average of the word embeddings
## TODO: Try pretrained text embeddings from transformer encoder
## TODO: CTRL-F, exit()
## TODO: Line 407

## argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-notes', required=True, help='run id notes to be printed out at the top of the argparser to help distinguish from other runs.')
parser.add_argument('-outDir', required=True, help='directory where output files will be written to')
parser.add_argument('-dataDir', required=True, help='directory where data files will be read in from')
parser.add_argument('-h5Dir', required=True, help='weiqius directory')
args = parser.parse_args()

## global variables
outdir, dataDir, h5Dir = args.outDir, args.dataDir, args.h5Dir
train_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'train' in f])
val_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'val' in f])
test_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'test' in f])
vocab_fnames = sorted([join(dataDir, f) for f in listdir(dataDir) if 'vocab' in f])
h5_fnames = sorted([join(h5Dir, f) for f in listdir(h5Dir) if '_board_inputs' in f])

## sanity checks
print('\n---argparser---:')
for arg in vars(args):
	print(arg, getattr(args, arg), '\t', type(arg))

print('\n---weiqius fnames---')
print(f'len: {len(h5_fnames)}')
for f in h5_fnames:
	print(f)

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
train_boards_fname, train_text_fname, train_choices_fname, train_choices_pkl_fname = train_fnames[2], train_fnames[4], train_fnames[1], train_fnames[0]
val_boards_fname, val_text_fname, val_choices_fname, val_choices_pkl_fname = val_fnames[2], val_fnames[4], val_fnames[1], val_fnames[0]
test_boards_fname, test_text_fname, test_choices_fname, test_choices_pkl_fname = test_fnames[2], test_fnames[4], test_fnames[1], test_fnames[0]
train_h5_fname, val_h5_fname, test_h5_fname = h5_fnames[1], h5_fnames[2], h5_fnames[0]
vocab_fname = vocab_fnames[0]

def get_training_sets(session, model):

		print('\n---TRAIN EXAMPLES---')
		train_board_feature_matrix = create_pretrained_board_feature_matrix('train', session, model, train_boards_fname, train_h5_fname) ## 'numpy.ndarray', <class 'sklearn.feature_extraction._dict_vectorizer.DictVectorizer'>
		pos_train_text_feature_matrix, neg_train_text_feature_matrix  = create_pretrained_text_feature_matrix(train_choices_pkl_fname, train_text_fname, vocab_fname) ## <class 'numpy.ndarray'>
		X_train, y_train = get_examples([train_board_feature_matrix], [pos_train_text_feature_matrix], [neg_train_text_feature_matrix], [train_choices_fname]) ##<class 'numpy.ndarray'>, <class 'numpy.ndarray'>
		file_obja, file_objb = open('data_pretrained_numpy/X_train.pkl', 'wb'), open('data_pretrained_numpy/y_train.pkl', 'wb')
		pkl.dump(X_train, file_obja)
		pkl.dump(y_train, file_objb)


		print('\n---VALIDATION EXAMPLES---')
		val_board_feature_matrix = create_pretrained_board_feature_matrix('val', session, model, val_boards_fname, val_h5_fname)
		pos_val_text_feature_matrix, neg_val_text_feature_matrix  = create_pretrained_text_feature_matrix(val_choices_pkl_fname, val_text_fname, vocab_fname)
		X_val,  y_val = get_examples([val_board_feature_matrix], [pos_val_text_feature_matrix], [neg_val_text_feature_matrix], [val_choices_fname])
		file_objc, file_objd = open('data_pretrained_numpy/X_val.pkl', 'wb'), open('data_pretrained_numpy/y_val.pkl', 'wb')
		pkl.dump(X_val, file_objc)
		pkl.dump(y_val, file_objd)


		print('\n---TEST EXAMPLES---')
		test_board_feature_matrix = create_pretrained_board_feature_matrix('test', session, model, test_boards_fname, test_h5_fname)
		pos_test_text_feature_matrix, neg_test_text_feature_matrix  = create_pretrained_text_feature_matrix(test_choices_pkl_fname, test_text_fname, vocab_fname)
		X_test,  y_test = get_examples([test_board_feature_matrix], [pos_test_text_feature_matrix], [neg_test_text_feature_matrix], [test_choices_fname])
		file_obje, file_objf = open('data_pretrained_numpy/X_test.pkl', 'wb'), open('data_pretrained_numpy/y_test.pkl', 'wb')
		pkl.dump(X_test, file_obje)
		pkl.dump(y_test, file_objf)


		print('\n---TRAIN+VALIDATION COMBINED EXAMPLES FOR AUTOGRID SEARCH---')
		X_comb, y_comb = get_examples([train_board_feature_matrix,val_board_feature_matrix], [pos_train_text_feature_matrix,pos_val_text_feature_matrix], [neg_train_text_feature_matrix,neg_val_text_feature_matrix], [train_choices_fname,val_choices_fname])
		file_objg, file_objh = open('data_pretrained_numpy/X_comb.pkl', 'wb'), open('data_pretrained_numpy/y_comb.pkl', 'wb')
		pkl.dump(X_comb, file_objg)
		pkl.dump(y_comb, file_objh)

		return X_train, y_train, X_val, y_val, X_comb, y_comb, X_test, y_test


def create_pretrained_board_feature_matrix(split, session, model, fname, h5_fname):

	start = time.time()
	# dict_keys(['boards', 'steps'])
	# X = DictVectorizer(dtype=np.float32, sparse=False)

	print(f'h5_fname: {h5_fname}')

	# generators for faster processing, lazy loading for memory efficiency
	with h5py.File(h5_fname, 'r') as hf:
		bin_input_datas = hf.get('bin_input_datas')
		global_input_datas = hf.get('global_input_datas')
		bin_input_datas = np.array(bin_input_datas) # 'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas
		global_input_datas = np.array(global_input_datas)
		print("Board's shape: ", bin_input_datas.shape) # Boards shape (209566, 361, 22)
		bin_iter = iter(bin_input_datas[:(bin_input_datas.shape[0]/10)])
		global_iter = iter(global_input_datas[:(bin_input_datas.shape[0]/10)])


	# board_data, color_data = next(lazy_pickle_load(fname)), next(lazy_pickle_load(fname)) 
	# batched_boards, batched_colors = np.array([x[3] for x in board_data]), np.array([x[2] for x in color_data])
	all_katago_boards = []
	for bin_batch, global_batch in progress_in_board_batches('current board', bin_iter, global_iter, bin_input_datas.shape[0],128):
		batched_katago_boards = katago.extract_intermediate.fetch_output_batch_with_bin_input(session, model, bin_batch, global_batch)
		all_katago_boards.extend(batched_katago_boards)
	# katago_boards = katago.extract_intermediate_optimized.extract_features_batch(session, model, batched_boards, batched_colors) ## katago_board['trunk'] is <class 'numpy.ndarray'> shape (19,19,256)
	# print(f'len(all_katago_boards): {len(all_katago_boards)} \ntype(all_katago_boards[0]): {type(all_katago_boards[0])} \ntype(all_katago_boards): {type(all_katago_boards)}') ## katago_boards.shape: (20676, 19, 19, 128)
	bin_input_datas = None
	global_input_datas = None
	all_katago_boards = np.array(all_katago_boards) #np.concatenate(all_katago_boards,axis=0)
	flattened_all_katago_boards = all_katago_boards.reshape(all_katago_boards.shape[0],-1,all_katago_boards.shape[3]) # flattened_all_katago_boards: (20676, 361, 128)
	# print(f'type(flattened_all_katago_boards): {type(flattened_all_katago_boards)} \nflattened_all_katago_boards: {flattened_all_katago_boards.shape}')
	# flattened_katago_boards = np.mean(katago_boards,axis=-1).reshape(katago_boards.shape[0], katago_boards.shape[1]*katago_boards.shape[2]) 
	# print(f'flattened_katago_boards.shape: {flattened_katago_boards.shape}') ##flattened_katago_boards.shape: (20676, 361)
	# temp = [dict(zip(range(len(board)),board)) for board in flattened_all_katago_boards]

	# X.fit(temp)
	# X_nparr = X.transform(temp)

	feature_time = time.time() - start
	# print(f'type(board_feature_matrix): {type(X_nparr)}') ##<class 'numpy.ndarray'> (20676, 361)
	# print(f'board_feature_matrix.shape: {X_nparr.shape}')
	print('Time elapsed making board feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(feature_time))))
	

	return flattened_all_katago_boards #X_nparr


def create_pretrained_text_feature_matrix(choices_pkl_fname, text_fname, vocab_fname, cutoff=5):

	start = time.time()

	comments, labels, vocab_size = prepare_comment(choices_pkl_fname, text_fname, vocab_fname, cutoff)
	labels = np.array(labels)[:(len(labels)/10)] #torch.tensor()
	true_temp_comments, neg_temp_comments = comments[np.where(labels==1)], comments[np.where(labels==0)]
	true_comments, neg_comments = true_temp_comments[:(true_temp_comments.shape[0]/10)], neg_temp_comments[:(neg_temp_comments.shape[0]/10)]
	true_temp_comments, neg_temp_comments = None, None
	# print(f'type(true_comments): {type(true_comments)}\ttype(neg_comments): {type(neg_comments)}')
	# print(f'true_comments.shape: {true_comments.shape}\tneg_comments.shape: {neg_comments.shape}')

	batch_size = 128
	ntokens = vocab_size # the size of vocabulary
	emsize = 200 # embedding dimension
	nhid = 100 # the dimension of the feedforward network pretrained_combine in nn.TransformerEncoder
	nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
	nhead = 2 # the number of heads in the multiheadattention models
	dropout = 0.2 # the dropout value

	text_model = load_model(ntokens, emsize, nhead, nhid, nlayers, dropout, device)

	## TODO: model = load_model()
	## TODO: example_features = extract_comment_features(text_model, example, batch_size, device)
	true_processed_feats, neg_processed_feats = [], []
	for i in range(0,true_comments.shape[0],batch_size): 
		true_batch, neg_batch = true_comments[i:i+batch_size], neg_comments[i:i+batch_size]
		true_example_features = extract_comment_features(text_model, true_batch, batch_size, device) #<class 'list'> (128, 100, 200)
		neg_example_features = extract_comment_features(text_model, neg_batch, batch_size, device) #<class 'list'> (128, 100, 200)
		true_processed_feats.extend(true_example_features)
		neg_processed_feats.extend(neg_example_features)
	# for true_batch, neg_batch in progress_in_text_batches('comments', true_comments_iter, neg_comments_iter, true_comments.shape[0],batch_size):
	#   true_example_features = extract_comment_features(text_model, true_batch, batch_size, device)
	#   neg_example_features = extract_comment_features(text_model, neg_batch, batch_size, device)
	#   print(f'type(true_example_features): {type(true_example_features)} \ntype(true_example_features[0]): {type(true_example_features[0])}')
	#   print(f'type(neg_example_features): {type(neg_example_features)} \ntype(neg_example_features[0]): {type(neg_example_features[0])}')
	#   true_processed_feats.extend(true_example_features)
	#   neg_processed_feats.extend(neg_example_features)

	print(f'-----text_feature_matrix----: {np.array(true_processed_feats).shape}')
	
	text_time = time.time() - start
	print('Time elapsed making text feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(text_time))))

	return true_processed_feats, neg_processed_feats #<class 'list'> , <class 'list'> 

def get_examples(boards_matrix, pos_text_matrix, neg_text_matrix, fnames):

	start = time.time()

	X, y, pca = [],[], PCA(n_components=100)


	for i, (board_feat_arr, pos_text_feat_arr, neg_text_feat_arr, fname) in enumerate(zip(boards_matrix, pos_text_matrix, neg_text_matrix, fnames)):
		# print(f'board_feat_arr.shape: {board_feat_arr.shape}')
		# print(f'len(pos_text_feat_arr): {len(pos_text_feat_arr)}\nlen(neg_text_feat_arr): {len(neg_text_feat_arr)}')
		# print(f'cur fname: {fname}\n')
		choices = next(lazy_load(fname))
		for line in choices:
			indices_str, pos_idx = line.split('\t')
			## append positive sample
			## Do PCA for board (20676, 361, 128)
			## Do PCA for text and text  --> (20676, 100,200)
			# board_feature.shape: (361, 128) 
			# text_feature.shape: (100, 200) 
			truncated_pos_text_feat_arr = np.array(pos_text_feat_arr[i])[:,:128] #(100,128)
			truncated_neg_text_feat_arr = np.array(neg_text_feat_arr[i])[:,:128] #(100,128)
			pos_board_dot_text = np.dot(board_feat_arr[int(pos_idx)],truncated_pos_text_feat_arr.T) # (361,100)
			neg_board_dot_text = np.dot(board_feat_arr[int(pos_idx)],truncated_pos_text_feat_arr.T) # (361,100)
			X.append(pos_board_dot_text.flatten()) # (36100,)
			y.append(1)
			## append negative example, select the 1st negative example from the group
			X.append(neg_board_dot_text.flatten()) # (36100,)
			y.append(0)

	examples_time = time.time() - start
	X_nparr, y_nparr = pca.fit_transform(np.array(X)), np.array(y)
	print(f'X num_examples: {X_nparr.shape}\ty num_examples: {y_nparr.shape}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return X_nparr, y_nparr 

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

		scores = ['precision', 'recall', 'f1']

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

	print(f'starting main with best parameters..')
	print(f'best parameters found: \n')
	print(best_params)
	print('\n')
	start = time.time()

	## logistic regression and fit X, y
	clf = LogisticRegression(C=best_params['C'], solver=best_params['solver'], penalty=best_params['penalty'], class_weight='balanced', random_state=0, max_iter=10000).fit(X, y)
	accuracy = clf.score(X_test, y_test)

	## get predictions using predict probabilities
	# temp = clf.predict_proba(X_test)[:,1]
	# print(f'type(temp): {type(temp)}')
	# print(f'temp[:100]: {temp[:100]}')
	# print('\n\n')
	# y_pred = np.zeros(temp.shape[0])
	# y_pred[np.where(temp>0.9)]=1
	# print(y_pred[:100])
	# print('\n\n')
	# print('y_pred.shape: ', y_pred.shape)

	## predictions using predict() method
	# y_pred = clf.predict(X_test)


	## print confusion matrix
	print(f'\n---Confusion Matrix---')
	cm = confusion_matrix(y_test, clf.predict(X_test)) #clf.predict(X_test)

	fig, ax = plt.subplots(figsize=(8, 8))
	ax.imshow(cm)
	ax.grid(False)
	ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
	ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
	ax.set_ylim(1.5, -0.5)
	for i in range(2):
		for j in range(2):
			ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
	plt.savefig("data_pretrained_numpy/confusion_matrix.png")

	## classification report
	print(f'\n---Classification Report---')
	print(classification_report(y_test, clf.predict(X_test)))


	main_with_best_parameters_time = time.time() - start
	print(f'end main with best parameters..')
	print('Time elapsed running main with best parameters only:\t%s\n' % (time.strftime("%H:%M:%S", time.gmtime(main_with_best_parameters_time))))

	return accuracy

################
### HELPER FUNCS
################

# def progress_in_batches(name, iterator, total_length, increment):
# 	count = 0
# 	while True:
# 		next_batch = list(islice(iterator, increment))
# 		if len(next_batch) > 0:
# 			count += len(next_batch)
# 			print('\nCreating features for {} batch... percentage processed {}'.format(name, (count/total_length)*100))
# 			yield next_batch
# 		else:
# 			break

def progress_in_board_batches(name, iterator1, iterator2, total_length, increment):
	count = 0
	while True:
		next_batch1 = list(islice(iterator1, increment))
		next_batch2 = list(islice(iterator2, increment))
		if len(next_batch1) > 0:
			count += len(next_batch1)
			print('Creating features for {} batch... percentage processed {}'.format(name, (count/total_length)*100))
			yield next_batch1, next_batch2
		else:
			break

def progress_in_text_batches(name, iterator1, iterator2, total_length, increment):
	count = 0
	while True:
		next_batch1 = islice(iterator1, increment)
		next_batch2 = islice(iterator2, increment)
		if len(next_batch1) > 0: #0
			count += len(next_batch1)
			if count % 640 == 0:
				print('Creating features for {} batch... percentage processed {}'.format(name, (count/total_length)*100))
			yield next_batch1, next_batch2
		else:
			break

def lazy_pickle_load(fname):
	file_obj = open(fname, "rb")
	contents = pkl.load(file_obj)['boards']
	file_obj.close()
	yield contents

def pickle_load(fname):
	file_obj = open(fname, "rb")
	contents = pkl.load(file_obj)
	file_obj.close()
	return contents

def lazy_load(fname):
	file_obj = open(fname,"r+")
	contents = file_obj.readlines()
	file_obj.close()
	yield contents


if __name__ == '__main__':

	start = time.time()

	saved_model_dir = "katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/" #"katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/" #"katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/"
	model, model_variables_prefix, model_config_json = katago.get_model(saved_model_dir)

	saver = tf.train.Saver(
		max_to_keep=10000,
		save_relative_paths=True,
	)

	tf.compat.v1.disable_eager_execution()

	with tf.Session() as session:
		saver.restore(session, model_variables_prefix)

		## get datasets
		X_train, y_train, X_val, y_val, X_comb, y_comb, X_test, y_test = get_training_sets(session, model)

		print('\n---Logistic Regression In Progress---')

		## Load Data
		best_params = {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}
		# X_train, y_train = pickle_load('data_pretrained_numpy/X_comb.pkl'), pickle_load('data_pretrained_numpy/y_comb.pkl')
		# X_test, y_test = pickle_load('data_pretrained_numpy/X_test.pkl'), pickle_load('data_pretrained_numpy/y_test.pkl') 
		# print(X_train.shape)
		# print(y_train.shape)
		# print(X_test.shape)
		# print(y_test.shape)

		## Run Main
		test_accuracy  = main(X_comb, y_comb, X_test, y_test, best_params)        
		print(f'test_accuracy: {test_accuracy}')

		## display results
		print('\n---Results---')
		for title, params, acc in zip(['*test*'], [best_params],[test_accuracy]):
			print(f'{title}\tbest_model: {params}\taccuracy: {acc}')
	
	main_time = time.time() - start
	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))




