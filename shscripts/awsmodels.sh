#!/bin/sh

source activate pytorch_p36

pip install --user torchkge
pip install --user tabulate

mkdir data/SVO-tensor-dataset/50
mkdir data/SVO-tensor-dataset/80

#python ~/weboftruth/corrupt.py

for ts in 100 80 50
do
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'DistMult' -p '~/weboftruth'
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'HolE' -p '~/weboftruth'
    python ~/weboftruth/weboftruth/wotmodels.py -e 2000 -emb 300 -ts $ts -m 'TransE' -p '~/weboftruth'
done
