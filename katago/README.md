# How to use KataGo features

Install requirements
```
pip install tensorflow==1.15
```

Assume you are already in the `katago` folder.

```
mkdir trained_models
cd trained_models
curl -O https://d3dndmfyhecmj0.cloudfront.net/g170/neuralnets/g170e-b20c256x2-s5303129600-d1228401921.zip
unzip g170e-b20c256x2-s5303129600-d1228401921.zip
cd ..
```

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

The example is also in `test/test_extract_katago_features.py`. The features is a dictionary that contains a lot of features extracted and an extra item named `trunk`. All others are features described in the appendix of their paper https://arxiv.org/abs/1902.10565. `trunk` is an intermediate layer of the model before the policy and value heads. You can choose to use 1) all other features 2) just `trunk`, 3) or all together.

You just need to call `tf.Session` once in the outside, and have multiple `extract_features` inside.