#!/bin/sh
#SBATCH --job-name=transe-gpu-lg
#SBATCH --output=/project2/jevans/aabir/weboftruth/logs/transe-lg-gpu.out
#SBATCH --error=/project2/jevans/aabir/weboftruth/logs/transe-lg-gpu.err
#SBATCH --mem=31GB
#SBATCH --time=32:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:3

#module load Anaconda3/5.3.0
#module load cuda/9.1

echo 'run started at ' $(date)

for ts in 50 80 100
do
    python /Users/aabir/Documents/research/weboftruth/weboftruth/wotmodels.py \
        -e 200 \
        -m 'TransE' \
        -lr 0.00005 \
        -dp /Users/aabir/Documents/research/datasets-knowledge-embedding \
        -ds 'KINSHIP' \
        -mp /Users/aabir/Documents/research/weboftruth/models \
        -ts $ts
done

echo echo 'run ended at ' $(date)
