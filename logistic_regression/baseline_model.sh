source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run34_AlphaGo
#$ -l h=nlpgrid13
#$ -l h_vmem=200G
python3 -u logistic_regression/baseline_model.py \
-notes 'elastic hyperparameters, 20K examples, max_iter=10K.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_elastic_hyperparams_20K_examples_max_iter_10K_run34.stdout' 2>&1 