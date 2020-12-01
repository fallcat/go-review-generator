import os
import pickle
from collections import defaultdict
from tqdm import tqdm

directory = r'processed'

train = defaultdict(list)
val = defaultdict(list)
test = defaultdict(list)
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".pkl"):
        with open(os.path.join(directory, filename), 'rb') as input_file:
            data = pickle.load(input_file)
        file_idx = int(filename.split('-')[0])
        if file_idx < 6300:
            train['boards'].extend(data['boards'])
            comments = [' '.join(comment.strip().split('\n')) for comment in data['comments']]
            train['comments'].extend(comments)
            train['steps'].extend([(file_idx, step) for step in data['steps']])
        elif file_idx < 7000:
            val['boards'].extend(data['boards'])
            comments = [' '.join(comment.strip().split('\n')) for comment in data['comments']]
            val['comments'].extend(comments)
            val['steps'].extend([(file_idx, step) for step in data['steps']])
        else:
            test['boards'].extend(data['boards'])
            comments = [' '.join(comment.strip().split('\n')) for comment in data['comments']]
            test['comments'].extend(comments)
            test['steps'].extend([(file_idx, step) for step in data['steps']])
    else:
        continue

print("train", len(train['boards']))
print("val", len(val['boards']))
print("test", len(test['boards']))

with open('data_splits/train.pkl', 'wb') as output_file:
    pickle.dump(train, output_file)

with open('data_splits/val.pkl', 'wb') as output_file:
    pickle.dump(val, output_file)

with open('data_splits/test.pkl', 'wb') as output_file:
    pickle.dump(test, output_file)

with open('data_splits/train_comments.txt', 'wt') as output_file:
    for comment in tqdm(train['comments']):
        output_file.write(comment + '\n')

with open('data_splits/val_comments.txt', 'wt') as output_file:
    for comment in tqdm(val['comments']):
        output_file.write(comment + '\n')

with open('data_splits/test_comments.txt', 'wt') as output_file:
    for comment in tqdm(test['comments']):
        output_file.write(comment + '\n')
