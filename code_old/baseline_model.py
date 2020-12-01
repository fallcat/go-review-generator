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

def get_examples(boards_mat, text_mat, fname):

	start = time.time()

	X, y = [],[]
	choices = next(lazy_load(fname))

	for line in choices:
		## split by tab first, then split by space
		indices_str, pos_idx = line.split('\t')
		cur_indices = [int(idx) for idx in indices_str.split()]
		## append positive example
		X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[int(pos_idx)])))
		y.append(1)
		## append negative example
		X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[random.choice(cur_indices)])))
		y.append(0)
		## end 50/50 positive to negative examples

		# cur_indices_iter = iter(cur_indices)
		# for i, cur_idx in enumerate(cur_indices_iter):
		# 	if i == int(pos_idx):
		# 		## match true board with true text and label 1
		# 		## TODO: make 9 more true examples to balance class imbalance
		# 		## TODO: try PCA first and then feed to logistic regression
		# 		X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[int(pos_idx)])))
		# 		y.append(1)
		# 	else:
		# 		## match true board with non-true text and label 0
		# 		## TODO: only append 4 out of the 9 negative examples
		# 		X.append(np.concatenate((boards_mat[int(pos_idx)],text_mat[cur_idx])))
		# 		y.append(0)
			

	examples_time = time.time() - start
	X_nparr, y_nparr = np.array(X), np.array(y)
	print(f'num_examples X, num_examples y: {X_nparr.shape,y_nparr.shape}')
	print('Time elapsed getting examples:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(examples_time))))

	return X_nparr, y_nparr

def manual_grid_search(title, X, y, X_test, y_test, fname):

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
	overall_accs, indvl_accs, best_params, max_acc_history = train(LogReg_models, X, y, X_test, y_test, regularization_vals, fname)	
	## display results
	display_results(title, best_params, regularization_vals, overall_accs)
	man_time = time.time() - start
	print('Time elapsed doing manual grid search:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(man_time))))

	return best_params, max_acc_history

def automatic_grid_search(X, y):

	# GRID SEARCH
	#The newton-cg and lbfgs solvers support only L2 regularization with primal formulation. 
	#The liblinear solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty.
	print(f'\nstarting auto parameter search..')
	start = time.time()
	# parameters = [{'penalty': ['l1'], 'solver':['liblinear'], 'C': [0.001,0.01,0.1,1,10], 'verbose': [0,1,2]}
	# 			{'penalty': ['l2'], 'solver':['lbfgs','liblinear','newton-cg','sag','saga'], 'C': [0.001,0.01,0.1,1,10], 'verbose': [0,1,2]}]
	parameters = {'C': [0.001,0.01,0.1]}
	clf = GridSearchCV(LogisticRegression(random_state=0, max_iter=5000), parameters) #, scoring='%s_macro' % score
	clf.fit(X, y)
	best_params = clf.best_params_
	parameter_search_time = time.time() - start
	print(f'end auto parameter search..')
	print('Time elapsed in doing an automatic parameter search:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(parameter_search_time))))

	return best_params


## ADDED HW4 STUB CODE
def train(models, X_train, y_train, X_test, y_test, params, fname):
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
	print(f'X_train.shape: {X_train.shape} \t y_train.shape: {y_train.shape}')
	print(f'X_test.shape: {X_test.shape} \t y_test.shape: {y_test.shape}')
	total_count = (X_test.shape[0]/10)
	print(f'total_count: {total_count}')
	overall_model_accuracies, indvl_model_accuracies = [],[]
	max_acc, best_params, max_acc_history = -float('inf'), None, [-float('inf')]

	# Load data
	choices = next(lazy_load(fname)) #generator obj
	choices_labels =  [int(line.split('\t')[1]) for line in choices]
	# len(choices_labels): 209566
	# choices_labels: 
	# [6, 1, 1, 8, 2, 7, 5, 6, 9, 1]

	for i, model in enumerate(models):

		# Logistic Regression
		model.fit(X_train, y_train)
		X_test_iter = iter(X_test)
		correct = 0

		temp = []
		for group_idx, group in enumerate(progress_of_predictions('current group of predictions', X_test_iter, X_test.shape[0],2)): ## this used to be groups of 10... can I just predict now instead of doing groups of 2?

			# Predict Class
			cur_probs = model.predict_proba(group)[:,1] #<class 'numpy.ndarray'> get probabilities its 1
			y_pred = np.argmax(cur_probs)
			try:
				if y_pred == choices_labels[group_idx]:
					correct += 1
					cur_acc = (100 * correct/total_count)
					temp.append(cur_acc)
				else:
					# print(f'\nround {group_idx} incorrect prediction (y_pred,y_true): {y_pred, choices_labels[group_idx]}')
					# print(f'current probabilities: \n{cur_probs}')
					cur_acc = (100 * correct/total_count)
					temp.append(cur_acc)
			except IndexError as e:
				print(e)
				print(group_idx)
				print(y_pred)
				print(len(choices_labels))

		# Accuracy 
		indvl_model_accuracies.append(temp)
		print(f'current model, num_correct: {correct} total: {total_count}')
		cur_models_overall_accuracy = (100 * correct/total_count)
		overall_model_accuracies.append(cur_models_overall_accuracy)

		# Save best parameters
		if cur_models_overall_accuracy > max_acc:
			max_acc = cur_models_overall_accuracy
			max_acc_history.append(max_acc)
			best_params = (i,params[i],max_acc)

	training_time = time.time() - start
	print(f'end logistic regression..') 
	print('Time elapsed manually training (manual grid search) :\t%s' % (time.strftime("%H:%M:%S", time.gmtime(training_time))))

	return overall_model_accuracies, indvl_model_accuracies, best_params, max_acc_history

def main(X, y, X_test, y_test, best_params, fname):

	start = time.time()
	
	# pen = best_params['penalty']
	# sol = best_params['solver']
	# c = best_params['C']
	# verb = best_params['verbose']

	# print(f'main() X.shape: {X.shape} \ny.shape: {y.shape} \nX_text.shape: {X_test.shape} \ny_test.shape: {y_test.shape} \nbest_params: {best_params} \nfname: {fname}')
	# Load data
	choices = next(lazy_load(fname))
	choices_labels =  [int(line.split('\t')[1]) for line in choices]
	print(f'len(choices_labels): {len(choices_labels)}')
	X_test_iter = iter(X_test)
	total_count, correct, accuracies = (X_test.shape[0]/10), 0, []
	print(f'total_count: {total_count}')
	# c = best_params[1] #format of best_params: (0, 0.001) == (model_index_ordinal, list of params)
					   #format of best_params: {'C': 0.001} 

	# Logistic Regression 
	# clf = LogisticRegression(penalty=best_params['penalty'], solver=best_params['solver'], C=best_params['C'], verbose=best_params['verbose'], random_state=0, max_iter=5000).fit(X, y)
	clf = LogisticRegression(C=0.001, random_state=0, max_iter=5000).fit(X, y)

	for group_idx, pred_group in enumerate(progress_of_predictions('current group', X_test_iter, X_test.shape[0],2)): ## see line 243 comment, same applies here

		# Predict Class
		cur_probs = clf.predict_proba(pred_group)[:,1]
		y_pred = np.argmax(cur_probs)
		try:
			if y_pred == choices_labels[group_idx]:
				correct += 1
				cur_acc = (correct/total_count)*100
				accuracies.append(cur_acc)
			else:
				# print(f'\nround {group_idx} incorrect prediction (y_pred,y_true): {y_pred, choices_labels[group_idx]}')
				# print(f'current probabilities: \n{cur_probs}')
				cur_acc = (correct/total_count)*100
				accuracies.append(cur_acc)
		except IndexError as e:
			print(e)
			print(group_idx)
			print(y_pred)
			print(len(choices_labels))


	# Accuracy
	print(f'current model, num_correct: {correct} total: {total_count}') 
	overall_accuracy = (100 * correct/total_count)

	training_time = time.time() - start
	print('Time elapsed in auto training:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(training_time))))

	return overall_accuracy, accuracies

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

## HELPER FUNCTION TO DISPLAY RESULTS FOR MANUAL GRID SEARCH
def display_results(title, best_params, params, accs):

	# ## HYPERPARAMETER SEARCH results
	# print(f'\n---{title} accuracy scores---')  
	# for (s,v,c), acc in zip(params,accs):
	# 	print(f'LogisticRegression Model solver={s} verbose={v} C={c} \taccuracy: {acc}')

	# print(f'---{title} best---')
	# print(f'LogisticRegression Model solver={best_params[1][0]} verbose={best_params[1][1]} C={best_params[1][2]}')
	# print(f'accuracy: {accs[best_params[0]]}')

	print(f'\n---{title} accuracy scores---')
	print(f'best_params: \n{best_params}') 
	print(f'params: \n{params}')
	print(f'accuracies: ') 
	for acc in accs:
		print(acc)

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
	X_train, y_train = get_examples(train_board_feature_matrix, train_text_feature_matrix, train_choices_fname) ##<class 'numpy.ndarray'>, <class 'numpy.ndarray'>

	print('\n---VALIDATION EXAMPLES---')
	val_board_feature_matrix = create_board_feature_matrix(val_boards_fname)
	val_text_feature_matrix  = create_text_feature_matrix(val_text_fname)
	X_val,  y_val = get_examples(val_board_feature_matrix, val_text_feature_matrix, val_choices_fname)

	print('\n---TEST EXAMPLES---')
	test_board_feature_matrix = create_board_feature_matrix(test_boards_fname)
	test_text_feature_matrix  = create_text_feature_matrix(test_text_fname)
	X_test,  y_test = get_examples(test_board_feature_matrix, test_text_feature_matrix, test_choices_fname)

	print('\n----------MANUAL GRID SEARCH----------')
	print('\n---train---')
	train_best_params, test_max_acc_history = manual_grid_search('train', X_train[:20000], y_train[:20000], X_train[:20000], y_train[:20000], train_choices_fname)
	print('\n---validation---')
	val_best_params, val_max_acc_history = manual_grid_search('val', X_train[:20000], y_train[:20000], X_val[:20000], y_val[:20000], val_choices_fname)
	print('\n---Logistic Regression---')
	test_overall_accuracy, test_accuracies = main(X_train[:20000], y_train[:20000], X_test, y_test, val_best_params, test_choices_fname)

	print('\n---Results Manual Grid search---')
	print(f'train_best_params: {train_best_params}\t{test_max_acc_history}')
	print(f'val_best_parameters: {val_best_params}\t{val_max_acc_history}')
	print(f'test_overall_accuracy: {test_overall_accuracy}')
	print(f'len of indvl test accs: {len(test_accuracies)}')
	for acc in test_accuracies[:min(20, len(test_accuracies))]:
		print(acc)

	print('\n----------AUTO GRID SEARCH----------')
	print('\n---train---')
	train_best_params = automatic_grid_search(X_train[:20000], y_train[:20000])
	print('\n---validation---')
	val_best_params = automatic_grid_search(X_val[:20000], y_val[:20000])
	print('\n---Logistic Regression---')
	test_overall_accuracy, test_accuracies = main(X_train[:20000], y_train[:20000], X_test, y_test, val_best_params, test_choices_fname)

	print('\n---Results Auto Grid search---')
	print(f'test best parameters: {train_best_params}')
	print(f'val best parameters: {val_best_params}')
	print(f'test_overall_accuracy: {test_overall_accuracy}')
	print(f'len of indvl accs: {len(test_accuracies)}')
	for acc in test_accuracies[:min(20, len(test_accuracies))]:
		print(acc)
	
	main_time = time.time() - start
	print('Time elapsed in main:\t%s' % (time.strftime("%H:%M:%S", time.gmtime(main_time))))


'''
## stats
print('\n\t---accuracy scores---')  
for model, acc in zip([1,5,10,50,100,500],accuracies):
print(f'RF {model} \taccuracy: {acc}')

'''