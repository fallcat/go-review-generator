source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N run29_AlphaGo
#$ -l h=nlpgrid14
#$ -l h_vmem=200G
python3 -u logistic_regression/baseline_model.py \
-notes 'l1/l2/elasticnet parameters and fancy report, max_iter=10K, 50/50 ratio, 2M examples subset, using model.predict() method, combined train+val example.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_elastic_l1_l2_params_fancy_report_2M_examples_run29.stdout' 2>&1 