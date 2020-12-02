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
from sklearn.metrics import accuracy_score, precision_score,recall_score,fbeta_score, confusion_matrix,roc_curve,auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
import nltk
nltk.download('stopwords') 
nltk.download('punkt')
#TODO: Line 163 and 202, change max_iter parameter for log reg model if necessary
#TODO: Line 137 and 208 CTRL-F [:20] to work with less data
#TODO: Line 204 indexing debuggin statement on choices_labels
'''Recommendations (meeting minutes)
	Try get something simple to work first (Don't do LSTM or Transformer for now)
	CNN for the go board, and just variations of CNN
	BoW 500 most frequent words, (too ambitiouspretrained embeddings for the words)
	1 pretrained board embeddings
	Board + move (concatenate)
	Board + Text (throw both into NN, concatenate)
'''

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
	## TODO: Find a go board embedding (find someone that's trained alpha go - mapping from board to stone)

	global stones_dict, args
	start = time.time()
	temp = []
	# dict_keys(['boards', 'steps'])
	X = DictVectorizer(dtype=np.float32, sparse=False) ## default np.float64
	data = next(lazy_pickle_load(fname)) 
	for board in data:
		## basically how many white stones, how many black stones is all the baseline can account for
		## how far along in the game are you
		flattened_board = board[3].flatten().tolist()
		modified_board = [board[0],board[1],stones_dict[board[2]]]+flattened_board
		temp.append(dict(zip(range(len(modified_board)),modified_board)))
	X.fit(temp)
	X_nparr = X.transform(temp)

	feature_time = time.time() - start
	print(f'board_feature_matrix.shape: {X_nparr.shape}')
	print('Time elapsed making board feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(feature_time))))

	return X_nparr


def create_text_feature_matrix(fname):

	global args

	start = time.time()
	## TODO: Try glove or pymagnitude word embedding, sum the average of the word embeddings

	matrix = CountVectorizer(dtype=np.float32, max_features=1000)
	data = next(lazy_load(fname))
	sentences = [sent.strip() for sent in data]
	X = matrix.fit_transform(sentences).toarray() ## transforms sentences to unigram features

	text_time = time.time() - start
	print(f'text_feature_matrix.shape: {X.shape}')
	print('Time elapsed making text feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(text_time))))

	return X

def get_examples(boards_mats, text_mats, fnames):

	start = time.time()

	X, y = [],[]

	for boards_mat, text_mat, fname in zip(boards_mats, text_mats,fnames):
		print(f'boards_mat.shape: {boards_mat.shape}')
		print(f'len(text_mat): {len(text_mat)}')
		print(f'cur fname: {fname}')
		choices = next(lazy_load(fname))
		for line in choices:
			## split by tab first, then split by space
			indices_str, pos_idx = line.split('\t')
			cur_indices = [int(idx) for idx in indices_str.split()]
			## append positive example
			X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[int(pos_idx)])))
			y.append(1)
			## append negative example, select a random choice from the 9 negative examples to choose from
			X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[random.choice(cur_indices)])))
			y.append(0)

	examples_time = time.time() - start
	X_nparr, y_nparr = np.array(X), np.array(y)
	print(f'X num_examples: {X_nparr.shape}\ty num_examples: {y_nparr.shape}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return X_nparr, y_nparr

def manual_grid_search(X, y, X_test, y_test):

	## Set grid search parameters
	start = time.time()
	regularization_vals = [0.001, 0.01, 0.1, 1.0] #[0.01, 0.001, 0.01, 0.1, 1.0, 10]
	# verbose_vals = [0,1,2]
	# solver = ['lbfgs', 'liblinear', 'newton-cg']
	# params = list(product(solver, verbose_vals, regularization_vals))
	
	## get model objects
	LogReg_models = get_LogisticRegression_models(regularization_vals)
	print(f'number of models: {len(LogReg_models)}')
	## train models
	accuracies, best_params = train(LogReg_models, X, y, X_test, y_test, regularization_vals)	

	man_time = time.time() - start
	print('Time elapsed doing manual grid search:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(man_time))))

	return accuracies, best_params

def automatic_grid_search(X, y):

	# GRID SEARCH
	#The newton-cg and lbfgs solvers support only L2 regularization with primal formulation. 
	#The liblinear solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty.
	print(f'starting auto parameter search..')
	start = time.time()
	# parameters = [{'penalty': ['l1'], 'solver':['liblinear'], 'C': [0.001,0.01,0.1,1,10], 'verbose': [0,1,2]}
	# 			{'penalty': ['l2'], 'solver':['lbfgs','liblinear','newton-cg','sag','saga'], 'C': [0.001,0.01,0.1,1,10], 'verbose': [0,1,2]}]
	parameters = {'C': [0.001,0.01,0.1]}
	clf = GridSearchCV(LogisticRegression(random_state=0, max_iter=5000), parameters) #, scoring='%s_macro' % score
	clf.fit(X, y)
	accuracy = clf.score(X,y)
	best_params = clf.best_params_
	parameter_search_time = time.time() - start
	print(f'end auto parameter search..')
	print('Time elapsed in doing an automatic parameter search:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(parameter_search_time))))

	return accuracy, best_params


## ADDED HW4 STUB CODE
def train(models, X, y, X_test, y_test, params):
	'''
		Trains several models and returns the test accuracy for each of them
		This is for the manual grid search
		Args:
		  models: list of model objects
		Returns:
		  overall_model_accuracies (float): overall accuracy of the model at the end of training
		  indvl_model_accuracies (float): list of progressive accuracies as each prediction is being made
		  best_params (list): list of the param combo that resulted in the best overall accuracy of the model
	'''

	start = time.time()
	print(f'X.shape: {X.shape} \t y.shape: {y.shape}')
	print(f'X_test.shape: {X_test.shape} \t y_test.shape: {y_test.shape}')

	accuracies = []
	max_acc, best_params = -float('inf'), None

	for i, model in enumerate(models):

		# Logistic Regression
		model.fit(X, y)
		cur_acc = model.score(X_test, y_test)
		accuracies.append(cur_acc)

		# Save best parameters
		if cur_acc > max_acc:
			max_acc = cur_acc
			best_params = (i,params[i],max_acc)

	training_time = time.time() - start
	print(f'end logistic regression..') 
	print('Time elapsed manually training (manual grid search) :\t%s' % (time.strftime("%H:%M:%S", time.gmtime(training_time))))

	return accuracies, best_params

def main(X, y, X_test, y_test, best_params):

	start = time.time()
	
	# pen = best_params['penalty']
	# sol = best_params['solver']
	# c = best_params['C']
	# verb = best_params['verbose']

	# Logistic Regression 
	# clf = LogisticRegression(penalty=best_params['penalty'], solver=best_params['solver'], C=best_params['C'], verbose=best_params['verbose'], random_state=0, max_iter=5000).fit(X, y)
	clf = LogisticRegression(C=0.001, random_state=0, max_iter=5000).fit(X, y)
	accuracy = clf.score(X_test, y_test)
	params = clf.get_params()

	training_time = time.time() - start
	print('Time elapsed in auto training:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(training_time))))

	return accuracy, params

################
### HELPER FUNCS
################

## ADDED HW4 STUB CODE
def get_LogisticRegression_models(params):
  '''
	  Creates model objects for Logistic Regression.
	  See the documentation in sklearn here:
	  https://scikit-learn.org/0.16/modules/generated/sklearn.linear_model.LogisticRegression.html
  '''
  
  # To complete: Create a list of objects for the classifier for each of the above "LogReg" types
  # LogReg_objs = [LogisticRegression(solver=s, verbose=v, C=c, random_state=0, max_iter=5000) for (s,v,c) in params]
  LogReg_objs = [LogisticRegression(C=c, random_state=0, max_iter=5000) for c in params]

  return LogReg_objs

def progress_of_predictions(name, iterator, total_length, increment):
	count = 0
	while True:
		next_batch = list(islice(iterator, increment))
		if len(next_batch) > 0:
			count += len(next_batch)
			if count % 100000 == 0:
				print(f'Generating predictions for {name}... percentage processed {(count/total_length)*100}')
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

	print('\n---TRAIN EXAMPLES---')
	train_board_feature_matrix = create_board_feature_matrix(train_boards_fname) ## 'numpy.ndarray', <class 'sklearn.feature_extraction._dict_vectorizer.DictVectorizer'>
	train_text_feature_matrix  = create_text_feature_matrix(train_text_fname) ## <class 'numpy.ndarray'>
	X_train, y_train = get_examples([train_board_feature_matrix], [train_text_feature_matrix], [train_choices_fname]) ##<class 'numpy.ndarray'>, <class 'numpy.ndarray'>

	print('\n---VALIDATION EXAMPLES---')
	val_board_feature_matrix = create_board_feature_matrix(val_boards_fname)
	val_text_feature_matrix  = create_text_feature_matrix(val_text_fname)
	X_val,  y_val = get_examples([val_board_feature_matrix], [val_text_feature_matrix], [val_choices_fname])

	print('\n---TRAIN+VALIDATION COMBINED EXAMPLES FOR AUTOGRID SEARCH---')
	X_comb,  y_comb= get_examples([train_board_feature_matrix,val_board_feature_matrix], [train_text_feature_matrix,val_text_feature_matrix], [train_choices_fname,val_choices_fname])

	print('\n---TEST EXAMPLES---')
	test_board_feature_matrix = create_board_feature_matrix(test_boards_fname)
	test_text_feature_matrix  = create_text_feature_matrix(test_text_fname)
	X_test,  y_test = get_examples([test_board_feature_matrix], [test_text_feature_matrix], [test_choices_fname])

	print('\n---Logistic Regression Auto Grid In Progress---')
	print('confirm num of X examples and num of y examples: ')
	print(f'X_comb.shape {X_comb.shape}\ty_comb.shape: {y_comb.shape}')
	accuracy, best_params = automatic_grid_search(X_comb, y_comb)
	test_accuracy, test_best_params = main(X_train, y_train, X_test, y_test, best_params)

	print('\n---Results Auto Grid search---')
	print(f'accuracy from auto search: {accuracy}\nbest_params from auto search: {best_params}')
	print(f'test_accuracy: {test_accuracy}, test_best_params: {test_best_params}')
	
	main_time = time.time() - start
	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))


'''
## stats
print('\n\t---accuracy scores---')  
for model, acc in zip([1,5,10,50,100,500],accuracies):
print(f'RF {model} \taccuracy: {acc}')

'''