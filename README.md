# Go review matcher

## Requirements
- Python 3.8

- pytorch 1.7.0

- CUDA 10.1

## Usage

Due to the storage limit, I only provide the model and weights here. Before you run any code, please put the data folder "data_splits_final" under "./transformer_encoder".

# 1. If you would like to generate the features first and use it as your input, directly run:
```
python test.py
```
The current code will generate features for validation set. You may need to remove the comment around line 21, line 119 and comment out around line 26 and line 155 to generate features for training set and test set. 

Again, due to the limit of memory, except the validation set, the output will be several small .pkl files. Those files are named "batch_\*" following the same order as the original data. Validation set will all be in one single file. Each file includes 2240 commment pairs. They are stored as a dictionary with two keys "features" and "labels". For example, we load the .pkl file to variable d. Then each comment pair has:
```
d["features"][0]=[comment_feature1,comment_feature2]
d["labels"][0]=[label1,label2]
```
"comment_feature1/2" are the features of comment 1 or 2 for the board index '0' with shape (100,100). "lable1/2" are the labels(0 or 1) for comment 1 or 2. 

You can decide how to use them in your model later. CAUTION: it may need a huge memory(100GB) and storage(100GB) to run.

# 2. If you would like to use trained encoder and embeb it to your model:

Please import "data_process.py" and "model.py" to your code first. The model you need is TransformerModel. In this case, you don't need the line 40 in "model.py". You only need to use the output of variable "feature". Be careful, the shape of features is (sentence_len, batch_size, hidden_dim), which is (100,64,100) in my model. You may would like to reshape it to (batch_size,sentence_len, hidden_dim) by "torch.transpose()".
