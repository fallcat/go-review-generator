# How to use KataGo features

Install requirements
```
pip install tensorflow==1.15
```

Sorry it's in `tensorflow` because the KataGo codebase used it. But the outputs are all in `numpy`.

Assume you are already in the `katago` folder.

```
mkdir trained_models
cd trained_models
curl -O https://d3dndmfyhecmj0.cloudfront.net/g170/neuralnets/g170e-b20c256x2-s5303129600-d1228401921.zip
unzip g170e-b20c256x2-s5303129600-d1228401921.zip
cd ..
```

If you need a smaller network, do this instead and the hidden state is dim 128:
```
curl -O https://d3dndmfyhecmj0.cloudfront.net/g170/neuralnets/g170e-b10c128-s1141046784-d204142634.zip
```


## Old way to get features (too slow)

When you use katago features, you can import it to your project and use `extract_features` as follows:

````
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import tensorflow as tf
import katago

board_arr1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
board_arr1 = np.array(board_arr1)
color1 = 'b'

saved_model_dir = "katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/"
model, model_variables_prefix, model_config_json = katago.get_model(saved_model_dir)

saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
)

with tf.Session() as session:
    saver.restore(session, model_variables_prefix)
    features1 = katago.extract_features(session, model, board_arr1, color1)
````

The example is also in `test/test_extract_katago_features.py`. The features is a dictionary that contains a lot of features extracted and an extra item named `trunk`. All others are features described in the appendix of their paper https://arxiv.org/abs/1902.10565. `trunk` is an intermediate layer of the model before the policy and value heads (with dimension (19, 19, 256) for this trained model). You can choose to use 1) all other features 2) just `trunk`, 3) or all together.

You just need to call `tf.Session` once in the outside, and have multiple `extract_features` inside.

To extract features by batch, do this instead:

```
features1 = katago.extract_intermediate_optimized.extract_features_batch(session, model, board_arrs, colors)
```

## Better way to get features (with pre-stored bin inputs and is fast!)

Download and unzip this file in your `data_splits_final` folder

```
https://drive.google.com/file/d/1BIRWI_5YlOxc6JTV9yRpYBBeVEYl92mm/view?usp=sharing
```

Install dependencies of h5py, which is much faster to use than storing with pickle.

```
pip install h5py
```

Example for getting data for validation set

```python
import os
import h5py
import numpy as np

data_dir = 'data_splits_final'
split = 'val'

h5_path = os.path.join(data_dir, split + '_board_inputs.h5')

with h5py.File(h5_path, 'r') as hf:
    bin_input_datas = hf.get('bin_input_datas')
    global_input_datas = hf.get('global_input_datas')
    bin_input_datas = np.array(bin_input_datas) # 'bin_input_datas': bin_input_datas, 'global_input_datas': global_input_datas
    global_input_datas = np.array(global_input_datas)
    print('Boards shape', bin_input_datas.shape)

```

These `bin_input_datas` and `global_input_datas` are for the whole dataset. You still need to use the `{split}.choices.pkl` file to get positive and negative examples. Once you have a batch of `bin_input_datas` and `global_input_datas`, you can easily get board features this way (replace the dots with your dataloader):

```python
import h5py
import numpy as np
import torch
import katago
import tensorflow as tf

saved_model_dir = "katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/"
board_model, model_variables_prefix, model_config_json = katago.get_model(saved_model_dir)

saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with tf.Session() as session:
    saver.restore(session, model_variables_prefix)

    bin_input_datas = ... # from some dataloader
    global_input_datas = ... # from some dataloader

    board_features = torch.tensor(
                        katago.extract_intermediate.fetch_output_batch_with_bin_input(session, board_model, bin_input_datas, global_input_datas)).to(device)
```
