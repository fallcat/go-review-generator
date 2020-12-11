source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run35_AlphaGo
#$ -l h=nlpgrid11
#$ -l h_vmem=200G
python3 -u logistic_regression/baseline_model.py \
-notes 'l2 best performing hyperparameters from test_acc: 0.3939444212351061, all 410K examples, max_iter=10K.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_l2_best_performing_all_410K_examples_max_iter_10K_run35.stdout' 2>&1 