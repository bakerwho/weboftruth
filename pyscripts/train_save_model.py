import pandas as pd
import pickle
import os
from os.path import join

import torchkge
import torch
from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader

from tqdm.autonotebook import tqdm

# fix this
wot_path = "~/weboftruth"
svo_data_path = join(wot_path, 'SVO-tensor-dataset')
dump_path = join(wot_path, 'dumps')
model_path = join(wot_path, 'models')

files = os.listdir(svo_data_path)
for f in files:
    if 'train' in f: tr_fp = f
    if 'valid' in f: val_fp = f
    if 'test' in f: test_fp = f

def read_data(tr_fp, val_fp, test_fp):
    tr_df = pd.read_csv(join(svo_data_path, tr_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    val_df = pd.read_csv(join(svo_data_path, val_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    test_df = pd.read_csv(join(svo_data_path, test_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    return tr_df, val_df, test_df





if __name__ == '__main__':
    tr_df, val_df, test_df = read_data(tr_fp, val_fp, test_fp)
