 import os
from os.path import join
import torchkge
from torchkge import models
import torch
import re
import pandas as pd

import weboftruth as wot

def get_file_names(ts=100):
    for f in os.listdir(wot.wotmodels.svo_paths[ts]):
        if 'train' in f: tr_fn = f
        if 'valid' in f: val_fn = f
        if 'test' in f: test_fn = f
    return tr_fn, val_fn, test_fn

def read_data(tr_fn, val_fn, test_fn, path):
    tr_df = pd.read_csv(join(path, tr_fn),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    val_df = pd.read_csv(join(path, val_fn),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    test_df = pd.read_csv(join(path, test_fn),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    return tr_df, val_df, test_df

def df_to_kg(df):
    assert df.shape[1] in [3, 4], 'Invalid DataFrame shape on axis 1'
    cols = ['from', 'rel', 'to']
    assert set(df.columns).issuperset(
                        set(cols)), f"DataFrame does not contain columns {cols}"
    return torchkge.data_structures.KnowledgeGraph(df[[cols]])

def load_model(model_folder, whichmodel='best_'):
    """ Loads a model from a .py file by initializing an empty model with
    appropriate parameters read from log.txt
    Inputs:
        model_folder: path containing log.txt and .pt models
        whichmodel: string to match to .pt name
    Usage:
        import weboftruth as wot
        from os.path import join
        wot.load_model(join(wot.models_path, 'TransE_01'))
    """
    with open(join(model_folder, 'log.txt'), 'r') as f:
        metadata = f.readlines()
    vars = parse_metadata(metadata)
    model_type = vars['model_type']
    if model_type in ['TransR', 'TransD', 'TorusE']:
        model = getattr(models, vars['model_type']+'Model'
                        )(ent_emb_dim=vars['ent_emb_dim'],
                        rel_emb_dim=vars['rel_emb_dim'],
                        n_entities=vars['n_entities'],
                        n_relations=vars['n_relations'])
    elif model_type in ['DistMult', 'HolE', 'TransE']:
        model = getattr(models, vars['model_type']+'Model'
                        )(emb_dim=vars['emb_dim'],
                        n_entities=vars['n_entities'],
                        n_relations=vars['n_relations'])
    else:
        raise ValueError("Not equipped to deal with this model")
    model_file = [x for x in os.listdir(model_folder
                            ) if whichmodel in x and '.pt' in x][0]
    model.load_state_dict(torch.load(join(model_folder, model_file),
                                    map_location=torch.device('cpu')))
    model.eval()
    return model

def parse_metadata(md):
    if not isinstance(md, list) and isinstance(md, str):
        md = md.split('\n')
    vars = {}
    for line in md:
        k, v = parseline(line)
        if k is None and v is None:
            continue
        vars[k] = v
    return vars

def parseline(line):
    kwds = dict(zip(['model_type', 'emb_dim', 'n_entities',
                     'n_relations', 'rel_emb_dim', 'ent_emb_dim'],
                     [str]+[int]*5))
    for kwd in kwds:
        tp = kwds[kwd]
        if kwd in line:
            return kwd, tp(line.replace(kwd, '').strip())
    if 'ent_emb' in line:
        return 'ent_emb_dim', int(re.findall(f'\d+', line)[0])
    if 'rel_emb' in line:
        return 'rel_emb_dim', int(re.findall(f'\d+', line)[0])
    return None, None


me = """variable    value
----------  ------------------------------------------------------------------
kg          <torchkge.data_structures.KnowledgeGraph object at 0x7f6fc7fedc50>
model_type  TransE
diss_type   L2
emb_dim     250
model       TransEModel(
              (ent_emb): Embedding(30492, 250)
              (rel_emb): Embedding(4538, 250)
            )
model_path  /project2/jevans/aabir/weboftruth/models/TransE_01
logfile     /project2/jevans/aabir/weboftruth/models/TransE_01/log.txt
lr          0.0004
n_epochs    100
b_size      32
n_entities  30492
n_relations 4538
"""
