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
from katago.extract_intermediate import extract_features
features = extract_features(board, color)
````