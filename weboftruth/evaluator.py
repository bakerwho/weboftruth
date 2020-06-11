import numpy as np
from weboftruth.utils import load_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd

import weboftruth as wot

paths = wot.svo_paths

def read_triples(filepath):
    """Read triples from filepath
    Input:
        filepath: file with format "{subject}\t{verb}\t{object}\t{bool}"
    """
    df = pd.read_csv(filepath, sep='\t')
    return df[['from', 'to', 'rel']].to_numpy(), df['true_positive'].to_numpy()

def get_vector_from_triple(triple, ent_vectors, rel_vectors):
    s, o, v = triple
    v1, v2, v3 = ent_vectors[s], rel_vectors[v], ent_vectors[o]
    return np.concatenate((v1, v2, v3), axis=0)

def get_svo_model_embeddings(filepath, emb_modelfolder, whichmodel='best_'):
    Xs, Ys = [], []
    model = load_model(emb_modelfolder, whichmodel=whichmodel)
    ent_vectors, rel_vectors = model.get_embeddings()
    sovs, Ys = read_triples(filepath)
    for sov in sovs:
        Xs.append(get_vector_from_triple(sov, ent_vectors, rel_vectors))
    return np.array(Xs), Ys

def get_svo_glove_embeddings(filepath, ent_glove, rel_glove):
    Xs, Ys = [], []
    sovs, Ys = read_triples(filepath)
    for sov in sovs:
        s, o, v = sov
        vector = np.concatenate((ent_glove[s], rel_glove[v], ent_glove[o]),
                                axis=0)
        Xs.append(vector)
    return np.array(Xs), Ys

def train_sklearnmodel(modelClass, Xs, Ys, **kwargs):
    model = modelClass(**kwargs)
    model.fit(Xs, Ys)
    Ypred = model.predict(Xs).astype('int32')
    acc = accuracy_score(Ypred, Ys.astype('int32'))
    print(f"Train accuracy on {model.__repr__()}: {acc*100} %")
    return model

def evaluate_model(model, Xs, Ys):
    Ypred = model.predict(Xs).astype('int32')
    acc = accuracy_score(Ypred, Ys.astype('int32'))
    print(f"Test accuracy on {model.__repr__()}: {acc*100} %")
    return acc

if __name__=='__main__':
    tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
    x_tr, y_tr = get_svo_model_embeddings(join(paths[50], tr_fn))
    x_te, y_te = get_svo_model_embeddings(join(paths[50], test_fn))
    for cls in [LinearRegression, Ridge, SVC]:
        model = train_sklearnmodel(cls, x_tr, y_tr)
        evaluate_model(model, x_te, y_te)

"""
import weboftruth as wot

from os.path import join

from sklearn.linear_model import LogisticRegression

#tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
#tr_fp = join(wot.svo_paths[50], tr_fn)
#test_fp = join(wot.svo_paths[50], test_fn)
#model_folder = wot.models_path

d_path = '/project2/jevans/aabir/weboftruth/data/SVO-tensor-dataset/50/'
train_fp, test_fp = [d_path+ v for v in ['svo_data_ts50_train_1000000.dat',
                                        'svo_data_ts50_test_50000.dat']]
model_folder = '/project2/jevans/aabir/weboftruth/models'

x_tr, y_tr = wot.get_svo_model_embeddings(train_fp, join(model_folder, 'DistMult_01'))

x_te, y_te = wot.get_svo_model_embeddings(test_fp,
                    join(model_folder, 'DistMult_01'))

model = wot.train_sklearnmodel(LogisticRegression, x_tr, y_tr)
wot.evaluate_model(model, x_te, y_te)
"""
