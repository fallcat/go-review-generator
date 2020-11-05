# Preprocessing

`mkdir data processed data_splits data_splits_final`

Put the go review data in the folder

```
bash preprocess.sh
python split_data.py
bash apply_bpe.sh
python remove_old_comments.py
python get_neg_samples.py
```
