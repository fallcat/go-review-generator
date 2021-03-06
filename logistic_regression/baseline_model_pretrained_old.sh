source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N newAlphaGo_run21
#$ -l h=nlpgrid11
#$ -l h_vmem=100G
python3 -u code_orig/baseline_model.py \
-notes '50/50 ratio, max_features=1000, max_iter=5000, 20000 examples subset, 5 regularization values, using model.predict() method, using autogrid search only combining training and validation example.' \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model_pretrained_auto_grid_search_5050ratio_200examples_run21.stdout' 2>&1 