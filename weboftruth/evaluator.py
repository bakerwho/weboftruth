import numpy as np
from weboftruth.utils import load_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

import weboftruth as wot

paths = wot.svo_paths

def read_triples(filepath):
    """Read triples from filepath
    Input:
        filepath: file with format "{subject}\t{verb}\t{object}\t{bool}"
    """
    with open(filepath, 'r') as f:
        for line in f:
            vals = [int(i) for i in line.split()]
            triple, istrue = vals[:3], vals[3]
            yield triple, istrue

def get_vector_from_triple(triple, ent_vectors, rel_vectors):
    s, v, o = triple
    v1, v2, v3 = ent_vectors[s], rel_vectors[v], ent_vectors[o]
    return np.concatenate((v1, v2, v3), axis=0)

def parse_data(filepath, emb_modelfolder, whichmodel='best_'):
    Xs, Ys = [], []
    model = load_model(emb_modelfolder, whichmodel=whichmodel)
    ent_vectors, rel_vectors = model.get_embeddings()
    for triple, istrue in read_triples(filepath):
        Xs.append(get_vector_from_triple(triple, ent_vectors, rel_vectors))
        Ys.append(int(istrue))
    return Xs, Ys

def train_linear_model(Xs, Ys):
    lrmodel = LinearRegression()
    lrmodel.fit(Xs, Ys)
    return lrmodel

def evaluate_linear_model(lrmodel, Xs, Ys):
    Y_pred = lrmodel.predict(Xs)

if __name__=='__main__':
    tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
    x_tr, y_tr = parse_data(join(paths[50], tr_fn))
    x_te, y_te = parse_data(join(paths[50], test_fn))

"""
import weboftruth as wot

from os.path import join

tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
x_tr, y_tr = wot.evaluator.parse_data(join(wot.svo_paths[50], tr_fn),
                                    join(wot.models_path, 'TransE_01'))

x_te, y_te = wot.evaluator.parse_data(join(paths[50], test_fn))
"""
