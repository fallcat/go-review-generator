# TODO: update virtenv depending on whether you are running on nlpgrid or nlpgpu
source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N MyoshuGo_run40
#$ -l h=nlpgrid11
#$ -l h_vmem=300G
python3 -u logistic_regression/baseline_model_pretrained_numpy.py \
-notes 'script: baseline_model_pretrained_numpy.py, PCA confirmed, doing the whole dataset now, predict(a), l2 best parameters found, on nlpgpu07, all examples.' \
-outDir 'out/' \
-dataDir 'data/' \
-h5Dir '/nlp/data/weiqiuy/go-review-matcher/data_splits_final/' \
> 'stdout/baseline_model_pretrained_numpy_l2_hyperparams_460K_examples_run40_final.stdout' 2>&1 