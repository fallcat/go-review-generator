# TODO: update virtenv depending on whether you are running on nlpgrid or nlpgpu
source ~/topgun/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run31_rev7_AlphaGo
#$ -l h=nlpgrid15
#$ -l h_vmem=100G
python3 -u logistic_regression/baseline_model_pretrained_numpy.py \
-notes 'script: baseline_model_pretrained_numpy.py, predict_prob(a), l2 best parameters found, on nlpgpu07, all examples.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_pretrained_numpy_l2_hyperparams_460K_examples_run31_rev7_final.stdout' 2>&1 