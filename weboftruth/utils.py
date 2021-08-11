from io import StringIO
from numpy import concatenate
import os
from os.path import join
import numpy as np
import pandas as pd
import re
import sys
import torchkge
from torchkge import models
import torch

import weboftruth as wot
from weboftruth._constants import *

def get_svo_file_names(ts=100, path='.', old=False, get_paths=False):
    svo_path = join(path, str(ts))
    if old:
        svo_path = svo_path+'_old'
    for f in os.listdir(svo_path):
        if get_paths:
            if 'train' in f: tr_fn = join(path, str(ts), f)
            if 'valid' in f: val_fn = join(path, str(ts), f)
            if 'test' in f: test_fn = join(path, str(ts), f)
        else:
            if 'train' in f: tr_fn = f
            if 'valid' in f: val_fn = f
            if 'test' in f: test_fn = f
    return tr_fn, val_fn, test_fn

def get_simonepri_filenames(datapath, dataset, id=True):
    id_str = 'as_id_' if id else 'as_text_'
    for f in os.listdir(join(datapath, dataset)):
        if id_str+'train' in f: tr_fn = f
        if id_str+'valid' in f: val_fn = f
        if id_str+'test' in f: test_fn = f
    return tr_fn, val_fn, test_fn

def read_data(tr_fn, val_fn, test_fn, path, explode_rels=False, rel_sep=None,
                colnames=['from', 'rel', 'to']):
    # consider deleting
    tr_df = pd.read_csv(join(path, tr_fn),
                       sep='\t', names=colnames)
    val_df = pd.read_csv(join(path, val_fn),
                       sep='\t', names=colnames)
    test_df = pd.read_csv(join(path, test_fn),
                       sep='\t', names=colnames)
    if explode_rels:
        if rel_sep is None:
            print("Using default separator: '/'")
            rel_sep = '/'
        tr_df, val_df, test_df = [explode_rel_column(df, colnames[1], rel_sep)
                                    for df in (tr_df, val_df, test_df)]

    return tr_df, val_df, test_df

def explode_rel_column(df, rel_colname='rel', rel_sep='/'):
    assert rel_colname in df.columns, f"'{rel_colname}' not in column names"
    rel_col2 = rel_colname+'_full'
    df.rename(columns={rel_colname: rel_col2}, inplace=True)
    df[rel_colname] = df[rel_col2].apply(lambda x: [i for i in x.split(
                                rel_sep) if len(i)>0])
    df = df.explode(rel_colname)
    return df

def get_simonepri_dataset_dfs(datapath, dataset):
    tr_fn, val_fn, test_fn = wot.utils.get_github_filenames(datapath,
                                dataset)
    #print(tr_fn)
    explode = 'FB15' in dataset.upper()
    dfs = wot.utils.read_data(tr_fn, val_fn, test_fn,
                                join(datapath, dataset),
                                explode_rels=explode,
                                )
    dfs = [df.drop('true_positive', axis=1
                ) if 'true_positive' in df.columns else df
                for df in dfs ]
    return dfs

def reshuffle_trte_split(dfs):
    print("Shuffling and resplitting train/val/test data")
    sizes = [df.shape[0] for df in dfs]
    full_df = pd.concat([dfs[0], dfs[1], dfs[2]])
    full_kg = wot.utils.df_to_kg(full_df)
    tr_kg, val_kg, test_kg = full_kg.split_kg(sizes=sizes)
    return tr_kg, val_kg, test_kg


def df_to_kg(df):
    cols = ['from', 'rel', 'to']
    assert set(df.columns)==set(cols), f"DataFrame does not contain columns {cols}"
    if isinstance(df, torchkge.KnowledgeGraph):
        print("Warning: input to utils.df_to_kg() was a KG")
        return df
    return torchkge.KnowledgeGraph(df[cols])

def kg_to_df(kg, skip_fail=True):
    i2e, i2r = ({v:k for k,v in dct.items()} for dct in (kg.ent2ix, kg.rel2ix))
    data = []
    failcount = {'entities':0, 'relations':0}
    for i, (h,t,r) in enumerate(kg):
        ent_h, ent_t, rel = None, None, None
        try:
            ent_h, ent_t = i2e[h], i2e[t]
        except KeyError as e:
            failcount['entities'] += 1
            if not skip_fail:
                raise e
        try:
            rel = i2r[r]
        except KeyError as e:
            failcount['relations'] += 1
            if not skip_fail:
                raise e
        if not any([x is None for x in [ent_h, ent_t, rel]]):
            data.append([ent_h, ent_t, rel])
    for k in ['entities', 'relations']:
        print(f"Could not retrieve {k} for {failcount[k]}/{i+1} facts")
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
    model_file = sorted([x for x in os.listdir(model_folder
                            ) if which in x and '.pt' in x], reverse=True)[0]
    print(f"loading model from {model_file}\n"\
        f"embedding dimension: {vars['emb_dim']}")
    model.load_state_dict(torch.load(join(model_folder, model_file),
                                    map_location=torch.device('cpu')))
    model.eval()
    print(model)
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

def read_evaluation_sov_triples(filepath, sep='\t'):
    """Read triples from filepath
    Input:
        filepath: file with format "{subject}\t{verb}\t{object}\t{bool}"
    """
    with open(filepath, 'r') as f:
        l = f.readline()
        if all(w in l for w in ('from', 'rel', 'to', 'true_positive')):
            df = pd.read_csv(filepath, sep=sep)
        else:
            df = pd.read_csv(filepath,
                names=['from', 'to', 'rel', 'true_positive'], sep=sep)
    return df[['from', 'to', 'rel']].to_numpy(), df['true_positive'].to_numpy()

class Embeddings():
    def __init__(self, model, kg):
        self.model = model
        self.kg = kg
        self.ent_vecs, self.rel_vecs = self.model.get_embeddings()
        self.ent_vec_d = self.ent_vecs.shape[1]
        self.rel_vec_d = self.rel_vecs.shape[1]

    def get_vector_from_sov_triple(self, s, o, rel):
        try:
            s_ind, o_ind = self.kg.ent2ix[s], self.kg.ent2ix[o]
            rel_ind = self.kg.rel2ix[rel]
            s_vec = self.ent_vecs[s_ind]
            o_vec = self.ent_vecs[o_ind]
            rel_vec = self.rel_vecs[rel_ind]
        except:
            return None

        """
        try:
            s_vec = self.ent_vecs[s_ind]
        except (KeyError, IndexError):
            s_vec = np.zeros((self.ent_vec_d,))
        try:
            o_vec = self.ent_vecs[o_ind]
        except:
            o_vec = np.zeros((self.ent_vec_d,))
        try:
            rel_vec = self.rel_vecs[rel_ind]
        except:
            rel_vec = np.zeros((self.rel_vec_d,))
        """
        return concatenate((s_vec, rel_vec, o_vec), axis=0)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def get_pos_eval_data(eval8, index=1,
            evaldatapath='/content/gdrive/MyDrive/thesiscode/evaldata/'):
  x_tr, y_tr = eval8.get_triplet_embeddings(
      f'{evaldatapath}/{index}_Positional_test_ts=0.5.dat')

  x_te, y_te = eval8.get_triplet_embeddings(
      f'{evaldatapath}/{index}_Positional_val_ts=0.5.dat')

  y_tr, y_te = np.array(y_tr), np.array(y_te)

  return x_tr, y_tr, x_te, y_te

def get_bern_eval_data(eval8, index=1,
            evaldatapath='/content/gdrive/MyDrive/thesiscode/evaldata/'):
  x_tr, y_tr = eval8.get_triplet_embeddings(
      f'{evaldatapath}/{index}_Bernoulli_test_ts=0.5.dat')

  x_te, y_te = eval8.get_triplet_embeddings(
      f'{evaldatapath}/{index}_Bernoulli_val_ts=0.5.dat')

  y_tr, y_te = np.array(y_tr), np.array(y_te)

  return x_tr, y_tr, x_te, y_te


def get_model_params_from_log(modelfolder, modelspath):
    if 'log.txt' in os.listdir(join(modelspath, modelfolder)):
        with open(join(modelspath, modelfolder, 'log.txt'), 'r') as f:
            logtxt = f.readlines()
        tr_sampler_line = [l for l in logtxt if 'Traintime sampler' in l][0]
        train_sampler = tr_sampler_line.split(':')[1].strip()
        c_sampler_line = [l for l in logtxt if 'Corruption sampler' in l][0]
        corrupt_sampler = c_sampler_line.split(':')[1].strip()
        ts_line = [l for l in logtxt if 'traints' in l][0]
        ts = [int(s) for s in ts_line.split() if s.isdigit()][0]
        return {'modelname': modelfolder, 'train_sampler': train_sampler,
              'corrupt_sampler': corrupt_sampler, 'truth_share': ts}
    else:
        raise ValueError(f'Invalid modelfolder {modelfolder}')
