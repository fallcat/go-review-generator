source ~/topgun/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run31_AlphaGo
#$ -l h=nlpgrid15
#$ -l h_vmem=100G
python3 -u logistic_regression/baseline_model_pretrained.py \
-notes 'simple test, C parameters only, 100 examples, max_iter=10K.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_pretrained_simple_test_C_hyperparams_100_examples_run31.stdout' 2>&1 