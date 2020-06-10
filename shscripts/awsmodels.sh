#!/bin/sh

source activate pytorch_p36

pip install --user torchkge
pip install --user tabulate

mkdir weboftruth/data/SVO-tensor-dataset/50
mkdir weboftruth/data/SVO-tensor-dataset/80

#python ~/weboftruth/corrupt.py

for ts in 100 80 50
do
    echo "running DistMult"
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'DistMult' -p '~/weboftruth'
    echo "running HolE"
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'HolE' -p '~/weboftruth'
    echo "running TransE"
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'TransE' -p '~/weboftruth'
done
