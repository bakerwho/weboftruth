import numpy as np
from weboftruth.utils import load_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from os.path import join
import pandas as pd

import weboftruth as wot
import argparse
#parser = argparse.ArgumentParser()

#parser.add_argument("-ts", "--truthshare", dest="ts", default=100,
#                      help="truth share of dataset", type=int)

#parser.add_argument("-m_folder", dest="emb_modelfolder", default="/home-nfs/tenzorok/weboftruth/models/TransE_02", help="model folder", type=str)

from weboftruth.utils import *

paths = wot.svo_paths

class Evaluator():
    def __init__(self, modelfolder, whichmodel='best_',
                emb_model=None, trainkg=None):
        if emb_model is None:
            emb_model = load_model(modelfolder, which=whichmodel)
        if trainkg is None:
            self.trainkg = load_kg(modelfolder, which='train')
        else:
            self.trainkg = trainkg
        self.embeddings = Embeddings(emb_model, self.trainkg)

    def set_pred_model(self, predmodelClass, **kwargs):
        self.pred_model = predmodelClass(**kwargs)

    def get_triples(self, filepath):
        sovs, Ys = read_triples(filepath)
        return sovs, Ys

    def get_svo_model_embeddings(self, filepath, sovs=None, Ys=None):
        Xs = []
        if sovs is None and Ys is None:
            sovs, Ys = self.get_triples(filepath)
        for s, o, v in sovs:
            Xs.append(self.embeddings.get_vector_from_triple(s, v, o))
        return np.array(Xs), Ys

    def get_svo_glove_embeddings(self, ent_glove, rel_glove,
                                filepath, sovs=None, Ys=None):
        # TODO: generalize to other embeddings
        Xs = []
        if sovs is None and Ys is None:
            sovs, Ys = self.get_triples(filepath)
        for s, o, v in sovs:
            vector = np.concatenate((ent_glove[s], rel_glove[v], ent_glove[o]),
                                    axis=0)
            Xs.append(vector)
        return np.array(Xs), Ys

    def train_pred_model(self, Xs, Ys, **kwargs):
        self.pred_model.fit(Xs, Ys)
        Ypred = self.pred_model.predict(Xs).astype('int32')
        acc = accuracy_score(Ypred, Ys.astype('int32'))
        print(f"Train accuracy on {self.pred_model.__repr__()}: {acc*100} %")
        return acc

    def evaluate_pred_model(self, Xs, Ys):
        Ypred = self.pred_model.predict(Xs).astype('int32')
        acc = accuracy_score(Ypred, Ys.astype('int32'))
        print(f"Test accuracy on {self.pred_model.__repr__()}: {acc*100} %")
        return acc

if __name__=='__main__':
    tr_fn, val_fn, test_fn = wot.utils.get_file_paths(50)
    evl8 = Evaluator(emb_modelfolder, whichmodel='')
    x_tr, y_tr = evl8.get_svo_model_embeddings(tr_fn)
    x_te, y_te = evl8.get_svo_model_embeddings(test_fn)
    for predmodel in [LinearRegression, Ridge, SVC]:
        evl8.set_pred_model(predmodel)
        evl8.train_pred_model(x_tr, y_tr)
        evl8.evaluate_pred_model(x_te, y_te)

"""
import weboftruth as wot

from os.path import join

from sklearn.linear_model import LogisticRegression

nearRegression, Ridge,#tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
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
