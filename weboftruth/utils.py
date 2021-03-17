import os
from os.path import join
import torchkge
from torchkge import models
import torch
import re
import pandas as pd
from numpy import concatenate

import weboftruth as wot
from weboftruth.constants import *

def get_file_names(ts=100, path='.', old=False, get_paths=False):
    svo_paths = {k:join(path, str(k)) for k in [100, 80, 50]}
    if old:
        svo_paths = {k:v+'_old' for k, v in svo_paths.items()}
    for f in os.listdir(svo_paths[ts]):
        if get_paths:
            if 'train' in f: tr_fn = join(path, str(ts), f)
            if 'valid' in f: val_fn = join(path, str(ts), f)
            if 'test' in f: test_fn = join(path, str(ts), f)
        else:
            if 'train' in f: tr_fn = f
            if 'valid' in f: val_fn = f
            if 'test' in f: test_fn = f
    return tr_fn, val_fn, test_fn

def read_data(tr_fn, val_fn, test_fn, path):
    try:
        tr_df = pd.read_csv(join(path, tr_fn),
                           sep='\t', names=['from', 'rel', 'to'])
        val_df = pd.read_csv(join(path, val_fn),
                           sep='\t', names=['from', 'rel', 'to'])
        test_df = pd.read_csv(join(path, test_fn),
                           sep='\t', names=['from', 'rel', 'to'])
    except:
        tr_df = pd.read_csv(join(path, tr_fn), sep='\t')
        val_df = pd.read_csv(join(path, val_fn), sep='\t')
        test_df = pd.read_csv(join(path, test_fn), sep='\t')
    return tr_df, val_df, test_df

def df_to_kg(df):
    assert df.shape[1]==3, 'Invalid DataFrame shape on axis 1'
    cols = ['from', 'rel', 'to']
    assert set(df.columns)==set(cols), f"DataFrame does not contain columns {cols}"
    return torchkge.data_structures.KnowledgeGraph(df)

def kg_to_df(kg):
    i2e, i2r = ({v:k for k,v in dct.items()} for dct in (kg.ent2ix, kg.rel2ix))
    data = []
    for hr, set_of_tails in kg.dict_of_tails.items():
        h, r = hr
        ent_h, rel = i2e[h], i2r[r]
        for tail in set_of_tails:
            ent_t = i2e[tail]
            data.append([ent_h, ent_t, rel])
    return pd.DataFrame(data, columns=['from', 'to', 'rel'])


def load_model(model_folder, which='best_'):
    """ Loads a model from a .py file by initializing an empty model with
    appropriate parameters read from log.txt
    Inputs:
        model_folder: path containing log.txt and .pt models
        which: string to match to .pt name
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
        model = getattr(models, model_type+'Model'
                        )(ent_emb_dim=vars['ent_emb_dim'],
                        rel_emb_dim=vars['rel_emb_dim'],
                        n_entities=vars['n_entities'],
                        n_relations=vars['n_relations'])
    elif model_type in ['DistMult', 'HolE', 'TransE']:
        model = getattr(models, model_type+'Model'
                        )(emb_dim=vars['emb_dim'],
                        n_entities=vars['n_entities'],
                        n_relations=vars['n_relations'])
    else:
        raise ValueError("Not equipped to deal with this model")
    model_file = [x for x in os.listdir(model_folder
                            ) if which in x and '.pt' in x][0]
    print(model_file)
    print(vars['emb_dim'])
    model.load_state_dict(torch.load(join(model_folder, model_file),
                                    map_location=torch.device('cpu')))
    model.eval()
    return model

def load_kg(modelfolder, which=''):
    kgfile = [f for f in os.listdir(modelfolder) if 'kg.csv' in f
                and which in f][0]
    df = pd.read_csv(join(modelfolder, kgfile))
    return torchkge.KnowledgeGraph(df)

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

def read_triples(filepath):
    """Read triples from filepath
    Input:
        filepath: file with format "{subject}\t{verb}\t{object}\t{bool}"
    """
    with open(filepath, 'r') as f:
        l = f.readline()
        if all(w in l for w in ('from', 'rel', 'to')):
            df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath,
                names=['from', 'to', 'rel', 'true_positive'], sep='\t')
    return df[['from', 'to', 'rel']].to_numpy(), df['true_positive'].to_numpy()

class Embeddings():
    def __init__(self, model, kg):
        self.model = model
        self.kg = kg
        self.ent_vecs, self.rel_vecs = self.model.get_embeddings()
        self.ent2ix, self.rel2ix = self.kg.ent2ix, self.kg.rel2ix

    def get_vector_from_triple(self, s, rel, o):
        try:
            s_ind, o_ind = self.ent2ix[s], self.ent2ix[o]
            rel_ind = self.rel2ix[rel]
        except (KeyError, IndexError):
            s_ind, rel_ind, o_ind = s, rel, o
        s_vec, o_vec = self.ent_vecs[s_ind], self.ent_vecs[o_ind]
        rel_vec = self.rel_vecs[rel_ind]
        return concatenate((s_vec, rel_vec, o_vec), axis=0)


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
