#!/bin/sh
#SBATCH --job-name=aabir-transe
#SBATCH --output=/project2/jevans/aabir/weboftruth/logs/transe-2.out
#SBATCH --error=/project2/jevans/aabir/weboftruth/logs/transe-2.err
#SBATCH --partition=broadwl
#SBATCH --mem=31GB
#SBATCH --time=32:00:00

module load Anaconda3/5.3.0

echo 'run started at ' $(date)
python /project2/jevans/aabir/weboftruth/pyscripts/train_save_model.py
echo echo 'run ended at ' $(date)
