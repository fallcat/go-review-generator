source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N ratiofiftyfifty_rev3
#$ -l h=nlpgrid11
#$ -l h_vmem=100G
python3 -u code/baseline_model.py \
-notes '50/50 ratio, max_features=1000, max_iter=5000, 20K examples subset, 5 regularization values, using model.predict_proba(x) method.' \
-outDir 'out/' \
-dataDir 'data/' \
-end_index 5 \
--FULL \
> 'stdout/baseline_model_automatic_and_manual_grid_search_5050ratio_run16.stdout' 2>&1 