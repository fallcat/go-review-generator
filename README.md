# Go review matcher

## Requirements
Python 3.6
sgfmill
tqdm
mosesdecoder (put in the root directory)
fastBPE (put in the root directory)

## Preprocessing

```
mkdir data processed data_splits data_splits_final
cd data
curl -O https://gtl.xmp.net/sgf/zip/INDEX.txt
curl -O https://gtl.xmp.net/sgf/zip/001-999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/1000-1999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/2000-2999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/3000-3999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/4000-4999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/5000-5999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/6000-6999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/7000-7999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/8000-8999-reviews.zip
curl -O https://gtl.xmp.net/sgf/zip/9000-10000-reviews.zip
find . -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;
cd ..

bash preprocess.sh
python split_data.py
bash apply_bpe.sh
python remove_old_comments.py
python get_neg_samples.py
```
