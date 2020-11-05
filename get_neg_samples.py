import os
import random
import pickle
from tqdm import tqdm
import numpy as np

random.seed(42)

dir = 'data_splits_final'
splits = ['val', 'test']
neg_samples = 9

for split in splits:
    with open(os.path.join(dir, f'{split}_comments.tok.32000.txt'), 'rt') as input_file:
        lines = len(list(input_file.readlines()))
        with open(os.path.join(dir, f'{split}.choices.txt'), 'wt') as output_file:
            choice_indices = []
            answers = []
            line_indices = list(range(lines))
            for idx in tqdm(line_indices):
                choices = [idx] + random.sample(line_indices[:idx] + line_indices[idx + 1:], neg_samples)
                random.shuffle(choices)
                choice_indices.append(choices)
                answer = choices.index(idx)
                answers.append(answer)

            random.shuffle(line_indices)
            choice_indices_shuffled = np.array(choice_indices)[line_indices]
            answers_shuffled = np.array(answers)[line_indices]
            for idx in tqdm(range(lines)):
                output_file.write(' '.join([str(n) for n in choice_indices_shuffled[idx]]) + '\t' + str(answers_shuffled[idx]) + '\n')
            with open(os.path.join(dir, f'{split}.choices.pkl'), 'wb') as output_file_bin:
                data = {'choice_indices': choice_indices_shuffled, 'answers': answers_shuffled}
                pickle.dump(data, output_file_bin)
