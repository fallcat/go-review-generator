source ~/annoy_normal/bin/activate
#!/bin/zsh
#$ -cwd
#$ -N alphaGo
#$ -l h=nlpgrid10
#$ -l h_vmem=50G
python3 -u code/baseline_model.py \
-outDir 'out/' \
-dataDir 'data/' \
> 'stdout/baseline_model.stdout' 2>&1 