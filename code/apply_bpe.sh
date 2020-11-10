TOOLS_PATH=$PWD

N_THREADS=16
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l en -no-escape -threads $N_THREADS"

eval "cat data_splits/train_comments.txt | $PREPROCESSING > data_splits/train_comments.tok.txt"
eval "cat data_splits/val_comments.txt | $PREPROCESSING > data_splits/val_comments.tok.txt"
eval "cat data_splits/test_comments.txt | $PREPROCESSING > data_splits/test_comments.tok.txt"

fastBPE/fast learnbpe 32000 data_splits/train_comments.tok.txt > data_splits/codes

fastBPE/fast applybpe data_splits/train_comments.tok.32000.txt data_splits/train_comments.tok.txt data_splits/codes

fastBPE/fast getvocab data_splits/train_comments.tok.32000.txt > data_splits/vocab.32000

fastBPE/fast applybpe data_splits/val_comments.tok.32000.txt data_splits/val_comments.tok.txt data_splits/codes data_splits/vocab.32000
fastBPE/fast applybpe data_splits/test_comments.tok.32000.txt data_splits/test_comments.tok.txt data_splits/codes data_splits/vocab.32000
