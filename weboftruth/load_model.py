import os
from os.path import join
import torchkge
from torchkge import models

"""
import sys
import argparse
parser = argparse.ArgumentParser()
"""

def load_model(model_folder, whichmodel=None):
    """ Loads a model from a .py file by initializing an empty model with
    appropriate parameters read from log.txt
    model_folder: path containing log.txt and .pt models
    whichmodel: string to match to .pt name
    """
    with open(join(model_folder), 'log.txt') as f:
        metadata = f.readlines()
    vars = parse_metadata(metadata)
    model_type = vars['model_type']
    if model_type in ['TransR', 'TransD', 'TorusE']:
        model = getattr(models, vars['model_type']+'Model')
            (ent_emb_dim=args['ent_emb_dim'], rel_emb_dim=args['rel_emb_dim'],
            n_entities=args['n_entities'], n_relations=args['n_relations'])
    elif model_type in ['DistMult', 'HolE', 'TransE']:
        model = getattr(models, vars['model_type']+'Model')
            (emb_dim=args['emb_dim'],
            n_entities=args['n_entities'], n_relations=args['n_relations'])
    else:
        raise ValueError("Not equipped to deal with this model")
    if whichmodel is None:
        whichmodel = 'best_'
    model_fn = [x for x in os.listdir(model_folder
                            ) if whichmodel in x and '.pt' in x][0]
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def parse_metadata(md):
    if not isinstance(md, list) and isinstance(md, str):
        md = md.split('\n')
    vars = {}
    for line in md:
        k, v = parseline(line, kwds)
        vars[k] = v
    return vars

def parseline(line):
    kwds = dict(zip(['model_type', 'emb_dim', 'n_entities',
                     'n_relations', 'rel_emb_dim', 'ent_emb_dim'],
                     [str]+[int]*5))
    for kwd in kwds:
        tp = kwds[kwd]
        if kwd in line:
            return kwd, tp(sent.replace(word, '').strip())
    if 'ent_emb' in line:
        return 'ent_emb_dim', int(re.findall(f'\d+', line)[0])
    if 'rel_emb' in line:
        return 'rel_emb_dim', int(re.findall(f'\d+', line)[0])


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
"""

# linear classifier

# mlp
