#!/bin/sh
#SBATCH --job-name=distmult-gpu-lg
#SBATCH --output=/project2/jevans/aabir/weboftruth/logs/distmult-lg-gpu.out
#SBATCH --error=/project2/jevans/aabir/weboftruth/logs/distmult-lg-gpu.err
#SBATCH --mem=31GB
#SBATCH --time=32:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:2

module load Anaconda3/5.3.0
module load cuda/9.1

echo 'run started at ' $(date)

for ts in 100 80 50
do
    python /project2/jevans/aabir/weboftruth/weboftruth/wotmodels.py -e 200 -m 'TransE' -lr 0.00005 -ts $ts
done

echo echo 'run ended at ' $(date)
