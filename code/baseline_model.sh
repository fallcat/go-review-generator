source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N alphaGo
#$ -l h=nlpgrid10
#$ -l h_vmem=100G
python3 -u code/baseline_model.py \
-outDir 'out/' \
-dataDir 'data/' \
-end_index 5 \
--FULL \
> 'stdout/baseline_model_job2.stdout' 2>&1 