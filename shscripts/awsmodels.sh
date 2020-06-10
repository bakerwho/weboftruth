#!/bin/sh

source activate pytorch_p36

pip install --user torchkge
pip install --user tabulate

mkdir weboftruth/data/SVO-tensor-dataset/50
mkdir weboftruth/data/SVO-tensor-dataset/80

#python ~/weboftruth/corrupt.py

for ts in 50 100 80
do
    echo "running DistMult"
    python ~/weboftruth/weboftruth/wotmodels.py -e 250 -emb 300 -ts $ts -m 'DistMult' -p '/home/ubuntu/weboftruth' -ve 10
    echo "running HolE"
    python /home/ubuntu/weboftruth/weboftruth/wotmodels.py -e 250 -emb 300 -ts $ts -m 'HolE' -p '/home/ubuntu/weboftruth' -ve 10
    echo "running TransE"
    python /home/ubuntu/weboftruth/weboftruth/wotmodels.py -e 250 -emb 300 -ts $ts -m 'TransE' -p '/home/ubuntu/weboftruth' -ve 10
done
