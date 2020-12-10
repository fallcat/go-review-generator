source ~/topgun/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run31_AlphaGo
#$ -l h=nlpgrid15
#$ -l h_vmem=100G
python3 -u logistic_regression/baseline_model_pretrained.py \
-notes 'script: baseline_model_pretrained.py, nlpgpu01 7, C parameters only.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_pretrained_simple_test_C_hyperparams_20K_examples_run31_rev5.stdout' 2>&1 