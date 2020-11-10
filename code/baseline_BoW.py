from os import listdir
from os.path import isfile, join
import random
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
nltk.download('stopwords') 
nltk.download('punkt')


parser = argparse.ArgumentParser(description='Processing list of files...')
parser.add_argument('-outDir', required=True, help='directory where output files will be written to')
parser.add_argument('-dataDir', required=True, help='directory where data files will be read in from')
parser.add_argument('-fname', required=True, help='directory where data files will be read in from')


'''global variables'''
train_fnames = [join(args.dataDir, f) for f in listdir(dataDir) if 'train' in f]
val_fnames = [join(args.dataDir, f) for f in listdir(dataDir) if 'val' in f]
test_fnames = [join(args.dataDir, f) for f in listdir(dataDir) if 'test' in f]

# Naive Bayes 

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


if __name__ == '__main__':


