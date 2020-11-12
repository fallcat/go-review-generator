from os import listdir
from os.path import isfile, join
import random
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse
import time
from itertools import product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from itertools import islice
import nltk
nltk.download('stopwords') 
nltk.download('punkt')
#TODO: Line 163 and 202, change max_iter parameter for log reg model if necessary

## argparser
parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help='directory where output files will be written to')
parser.add_argument('-dataDir', required=True, help='directory where data files will be read in from')
parser.add_argument('-end_index', type=int, required=False, help='if provided, will be used in conjuction with DEBUG flag to set small subset.')
parser.add_argument('--FULL', action='store_false', dest='DEBUG_flag', required=False, help='Enable script on full dataset')
parser.add_argument('--DEBUG', action='store_true', dest='DEBUG_flag', required=False, help='Enable script on small subset in order to debug')
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

	global stones_dict, args
	start = time.time()
	temp = []
	# dict_keys(['boards', 'steps'])
	X = DictVectorizer(dtype=np.float32, sparse=False) ## default np.float64
	file_obj = open(fname, "rb")
	data = pkl.load(file_obj)['boards']
	if args.DEBUG_flag:
		# print('\nrunning boards on small subset of dataset..')
		data_iter = iter(data[:args.end_index])
	else:
		# print('\nrunning board on full dataset..')
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

	global args

	start = time.time()

	matrix = CountVectorizer(dtype=np.float32, max_features=1000)

	file_obj = open(fname,"r+")
	data = file_obj.readlines()
	if args.DEBUG_flag:
		# print('\nrunning text on small subset of dataset..')
		data_iter = iter(data[:args.end_index])
	else:
		# print('\nrunning text on full dataset..')
		data_iter = iter(data)
	sentences = [sent.strip() for sent in data_iter]
	X = matrix.fit_transform(sentences).toarray() ## transforms sentences to unigram features

	text_time = time.time() - start
	print('\nTime elapsed making text feature matrix:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(text_time))))

	return X

def get_examples(boards_mat, text_mat, choices_fname):

	start = time.time()
	print(f'\nLine 125 boards_mat.shape: {boards_mat.shape} text_mat.shape: {text_mat.shape}')

	X, y = [],[]
	file_obj = open(choices_fname, "r+")
	choices = file_obj.read().splitlines()
	choices_iter = iter(choices)
	for line in choices_iter:
		cur_indices = [int(idx) for idx in line.split()]
		pos_idx = cur_indices[cur_indices[-1]]
		## positive examples
		X.append(np.concatenate((boards_mat[pos_idx],text_mat[pos_idx])))
		y.append(1)
		## negative examples
		cur_indices.remove(pos_idx)
		neg_iter = iter(cur_indices[:-1])
		for neg_idx in neg_iter:
			X.append(np.concatenate((boards_mat[neg_idx],text_mat[neg_idx])))
			y.append(0)

	examples_time = time.time() - start
	X_nparr, y_nparr = np.array(X), np.array(y)
	print(f'\nLine 146 X.shape, y.shape: {X_nparr.shape, y_nparr.shape}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return X_nparr, y_nparr

def main(X, y, X_test, y_test, best_params):
	'''
		Increase the number of iterations (max_iter) or scale the data as shown in:
		https://scikit-learn.org/stable/modules/preprocessing.html
		Please also refer to the documentation for alternative solver options:
		https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
		extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
	'''
	start = time.time()

	print(f'Line 161 X.shape: {X.shape} y.shape: {y.shape} X_text.shape: {X_test.shape} y_test.shape: {y_test.shape}')

	# Logistic Regression 
	clf = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], verbose=best_params['verbose'], random_state=0, max_iter=5000).fit(X, y)

	# Predict Class
	y_pred = clf.predict(X_test)

	# Accuracy 
	accuracy = accuracy_score(y_pred, y_test)

	training_time = time.time() - start 
	print('Time elapsed training :\t%s' % (time.strftime("%H:%M:%S", time.gmtime(training_time))))

	return accuracy


## ADDED HW4 STUB CODE
def get_LogisticRegression_models(param_combos):
  '''
	  Creates model objects for Logistic Regression.
	  See the documentation in sklearn here:
	  https://scikit-learn.org/0.16/modules/generated/sklearn.linear_model.LogisticRegression.html
  '''
  random_state = 0 # Do not change this random_state
  
  # To complete: Create a list of objects for the classifier for each of the above "LogReg" types
  LogReg_objs = [LogisticRegression(C=c, verbose=v, random_state=random_state, max_iter=5000) for c,v in param_combos]

  return LogReg_objs


## ADDED HW4 STUB CODE
def train(models, X_train, y_train, X_test, y_test, param_combos):
  """
  Trains several models and returns the test accuracy for each of them
  Args:
      models: list of model objects
  Returns:
      score (float): list of accuracies of the different fitted models on test set
  """

  # To complete: train and test each model in a for lop
  print(f'X_train.shape: {X_train.shape} \t y_train.shape: {y_train.shape}')
  print(f'X_test.shape: {X_test.shape} \t y_test.shape: {y_test.shape}')
  accuracies = []
  max_acc, best_C, best_verbose = -float('inf'), None, None

  for i, model in enumerate(models):
    ## train
    model.fit(X_train, y_train)
    ## score
    cur_acc = model.score(X_test, y_test)
    accuracies.append(cur_acc)
    if cur_acc > max_acc:
    	max_acc = cur_acc
    	best

  return accuracies

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

def display_results(param_combos, train_accs, val_accs):

	## HYPERPARAMETER SEARCH results
	print('\n\t---train accuracy scores---')  
	for (c,v), acc in zip(param_combos,train_accs):
		print(f'LogisticRegression Model C={c} verbose={v}\taccuracy: {acc}')

		print('\n\t---validation accuracy scores---')  
	for (c,v), acc in zip(param_combos,val_accs):
		print(f'LogisticRegression Model C={c} verbose={v}\taccuracy: {acc}')





if __name__ == '__main__':

	start = time.time()

	## TRAIN
	# get feature matrices
	train_board_feature_matrix = create_board_feature_matrix(train_boards_fname) ## 'numpy.ndarray', <class 'sklearn.feature_extraction._dict_vectorizer.DictVectorizer'>
	train_text_feature_matrix  = create_text_feature_matrix(train_text_fname) ## <class 'numpy.ndarray'>
	print(f'train_board_feature_matrix.shape: {train_board_feature_matrix.shape}\ntrain_text_feature_matrix.shape: {train_text_feature_matrix.shape}')
	# get training examples
	X_train, y_train = get_examples(train_board_feature_matrix, train_text_feature_matrix, train_choices_fname)
	X_train_shuffled, y_train_shuffled = shuffle(X_train,y_train)

	## VAL
	# get feature matrices
	val_board_feature_matrix = create_board_feature_matrix(val_boards_fname)
	val_text_feature_matrix  = create_text_feature_matrix(val_text_fname)
	print(f'val_board_feature_matrix.shape: {val_board_feature_matrix.shape}\nval_text_feature_matrix.shape: {val_text_feature_matrix.shape}')
	# get validation examples
	X_val,  y_val = get_examples(val_board_feature_matrix, val_text_feature_matrix, val_choices_fname)

	## TEST
	# get feature matrices
	test_board_feature_matrix = create_board_feature_matrix(test_boards_fname)
	test_text_feature_matrix  = create_text_feature_matrix(test_text_fname)
	print(f'test_board_feature_matrix.shape: {test_board_feature_matrix.shape}\ntest_text_feature_matrix.shape: {test_text_feature_matrix.shape}')
	# get test examples
	X_test,  y_test = get_examples(test_board_feature_matrix, test_text_feature_matrix, test_choices_fname)

	

	# ## MANUAL HYPERPARAMETER SEARCH
	# regularization_vals = [0.001, 0.01, 0.1, 1.0]
	# verbose_vals = [0,1,2]
	# param_combos = product(regularization_vals, verbose_vals)
	# LogReg_models = get_LogisticRegression_models(param_combos)
	# train_accs = train(LogReg_models, X_train_shuffled, y_train_shuffled, X_train, y_train, param_combos)
	# val_accs = train(LogReg_models, X_train_shuffled, y_train_shuffled, X_val, y_val, param_combos)
	# display_results(param_combos,train_accs, val_accs)

	## GRID SEARCH
	parameters = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000], 'verbose': [0,1,2]}
	clf = GridSearchCV(LogisticRegression(random_state=0, max_iter=10000), parameters) #, scoring='%s_macro' % score
	clf.fit(X_train_shuffled, y_train_shuffled)
	best_params = clf.best_params_
	print(f'best params based off GridSearchCV: {best_params}')


	## MAIN USING THE BEST PARAMETERS
	train_acc = main(X_train_shuffled, y_train_shuffled, X_train, y_train, best_params)
	val_acc = main(X_train_shuffled, y_train_shuffled, X_val, y_val, best_params)
	# test scores
	best_model = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], verbose=best_params['verbose'], random_state=0, max_iter=5000).fit(X_train, y_train)
	best_acc = best_model.score(X_test,y_test)

	main_time = time.time() - start

	print('\n---baseline stats---')
	print(f'train accuracy:\t{train_acc}')
	print(f'val accuracy:\t{val_acc}')
	print(f'best accuracy:\t{best_acc}')

	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))



