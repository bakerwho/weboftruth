import numpy as np
from weboftruth.utils import load_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from os.path import join
import pandas as pd

import weboftruth as wot
import argparse
#parser = argparse.ArgumentParser()

#parser.add_argument("-ts", "--truthshare", dest="ts", default=100,
#                      help="truth share of dataset", type=int)

#parser.add_argument("-m_folder", dest="emb_modelfolder", default="/home-nfs/tenzorok/weboftruth/models/TransE_02", help="model folder", type=str)

from weboftruth.utils import *

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

    def get_triples(self, filepath, sep='\t'):
        sovs, Ys = wot.utils.read_evaluation_sov_triples(filepath, sep=sep)
        return sovs, Ys

    def get_triplet_embeddings(self, filepath, sovs=None, Ys=None, sep='\t'):
        Xs = []
        skipcount = 0
        if sovs is None or Ys is None:
            sovs, Ys = self.get_triples(filepath, sep=sep)
        Ys2 = []
        for i, (s, o, v) in enumerate(sovs):
            vec = self.embeddings.get_vector_from_sov_triple(s, o, v)
            if vec is None:
                # try swapping o and v
                o, v = v, o
            vec = self.embeddings.get_vector_from_sov_triple(s, o, v)
            if vec is None:
                skipcount += 1
                continue
            Xs.append(vec)
            Ys2.append(Ys[i])
        print(f"Could not retrieve embeddings for {skipcount}/{i+1} triples")
        return np.array(Xs), Ys2

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
        self.pred_model.fit(Xs, Ys, **kwargs)
        Ypred = self.pred_model.predict(Xs).astype('int32')
        acc = accuracy_score(Ypred, Ys.astype('int32'))
        print(f"Train accuracy on {self.pred_model.__repr__()}: {acc*100} %")
        return acc

    def evaluate_pred_model(self, Xs, Ys, **kwargs):
        Ypred = self.pred_model.predict(Xs, **kwargs).astype('int32')
        acc = accuracy_score(Ypred, Ys.astype('int32'))
        print(f"Test accuracy on {self.pred_model.__repr__()}: {acc*100} %")
        return acc

def link_prediction_evaluation(model, test_kg, b_size=1, k=10):
    assert isinstance(model, torchkge.models.interfaces.Model)
    lp_eval8 = torchkge.evaluation.LinkPredictionEvaluator(model, test_kg)
    lp_eval8.evaluate(b_size)
    lp_eval8.print_results()
    h, filt_h = lp_eval8.hit_at_k(k)
    mr, filt_mr = lp_eval8.mean_rank()
    mrr, filt_mrr = lp_eval8.mrr()

    lp_results = {'lp_mrr':mrr, 'lp_filt_mrr':filt_mrr,
                  'lp_mr':mr, 'lp_filt_mr':filt_mr,
                  'lp_hit_k10':h, 'lp_filt_hit_k10':filt_h}

    return lp_results


def binary_classifiers_evaluation(modelfolder, trainkg, modelspath,
                                  whichmodel='best_', sampling='Positional'):
    eval8 = wot.evaluate.Evaluator(join(modelspath, modelfolder),
                               whichmodel,
                               trainkg=trainkg)
    if sampling == 'Positional':
      x_tr, y_tr, x_te, y_te = get_pos_eval_data(eval8)
    elif sampling == 'Bernoulli':
      x_tr, y_tr, x_te, y_te = get_bern_eval_data(eval8)
    else:
      raise ValueError(f'Invalid sampling: {sampling}')

    print(x_tr.shape, x_te.shape)

    results_dict = {}

    for name, predmodel in zip(['Ridge', 'LogisticRegression', 'MLPClassifier'],
                               [Ridge, LogisticRegression, MLPClassifier]):
        kwargs_dict = {'set':{}, 'train':{}, 'evaluate':{}}
        if predmodel == LogisticRegression:
            kwargs_dict['set']['solver'] = 'newton-cg'
        elif predmodel == Ridge:
            kwargs_dict['set']['max_iter'] = 10000
        eval8.set_pred_model(predmodel, **kwargs_dict['set'])
        train_acc = eval8.train_pred_model(x_tr, y_tr, **kwargs_dict['train'])
        test_acc = eval8.evaluate_pred_model(x_te, y_te, **kwargs_dict['evaluate'])
        results_dict[f'{name}_train_acc'] = train_acc
        results_dict[f'{name}_test_acc'] = test_acc
    return results_dict

if __name__=='__main__':
    ts=50
    w_model = 'best_'
    emb_modelfolder = './models/TransE_01'
    tr_fn, val_fn, test_fn = wot.utils.get_file_names(ts,
                        './data/SVO-tensor-dataset',
                        old=True, get_paths=True)
    print(tr_fn, val_fn, test_fn)
    eval8 = Evaluator(emb_modelfolder, whichmodel='best_')
    x_tr, y_tr = eval8.get_triplet_embeddings(tr_fn)
    x_te, y_te = eval8.get_triplet_embeddings(test_fn)
    for predmodel in [Ridge, LogisticRegression, MLPClassifier]:
        kwargs_dict = {'set':{}, 'train':{}, 'evaluate':{}}
        if predmodel == LogisticRegression:
            modelkwargs['solver'] = 'newton-cg'
        elif predmodel == Ridge:
            trainkwargs['max_iter'] = 10000
        eval8.set_pred_model(predmodel, **kwargs_dict['set'])
        eval8.train_pred_model(x_tr, y_tr, **kwargs_dict['train'])
        eval8.evaluate_pred_model(x_te, y_te, **kwargs_dict['evaluate'])


def triplet_classification_evaluation(model, tr_kg, val_kg, test_kg, b_size=1):
    tc_eval8 = TripletClassificationEvaluator(model, val_kg, test_kg)
    tc_eval8.sampler = torchkge.sampling.PositionalNegativeSampler(tr_kg, val_kg, test_kg)
    # copying evaluation code here
    r_idx = tc_eval8.kg_val.relations
    neg_heads, neg_tails = tc_eval8.sampler.corrupt_kg(b_size, tc_eval8.is_cuda,
                                                    which='val')
    neg_scores = tc_eval8.get_scores(neg_heads, neg_tails, r_idx, b_size)
    tc_eval8.thresholds = torch.zeros(tc_eval8.kg_val.n_rel)

    for i in range(tc_eval8.kg_val.n_rel):
        mask = (r_idx == i).bool()
        if mask.sum() > 0:
            tc_eval8.thresholds[i] = neg_scores[mask].max()
        else:
            tc_eval8.thresholds[i] = neg_scores.max()

    tc_eval8.evaluated = True
    tc_eval8.thresholds.detach_()
    print('tc_eval8.evaluate() code complete')

    r_idx = tc_eval8.kg_test.relations

    neg_heads, neg_tails = tc_eval8.sampler.corrupt_kg(b_size,
                                                    tc_eval8.is_cuda,
                                                    which='test')
    scores = tc_eval8.get_scores(tc_eval8.kg_test.head_idx,
                              tc_eval8.kg_test.tail_idx,
                              r_idx,
                              b_size)
    print('positive scores calculated')
    print('failed after positive scores: 3 times')
    print(tc_eval8.kg_test.head_idx.shape, tc_eval8.kg_test.tail_idx.shape, r_idx.shape)
    print(neg_heads.shape, neg_tails.shape, r_idx.shape)
    print(tc_eval8.kg_test.head_idx[:5], neg_heads[:5])
    neg_scores = tc_eval8.get_scores(neg_heads, neg_tails, r_idx, b_size)
    print('negative scores calculated')

    thresholds = tc_eval8.thresholds.cpu()
    scores = scores.cpu()
    neg_scores = neg_scores.cpu()

    true_pos = (scores > thresholds[r_idx])
    true_neg = (neg_scores < thresholds[r_idx])

    print('sums calculated')
    tc_acc = (true_pos.sum().item() + true_neg.sum().item()) / (2 * tc_eval8.kg_test.n_facts)

    # done copying
    print(f'Accuracy on Triplet Classification on test set: {tc_acc}')

    return {'tc_test_acc':tc_acc}

def eval_all_models_at_path(modelspath, train_kg, val_kg, test_kg):
    tc_result_dict = defaultdict(list)
    lp_result_dict = defaultdict(list)
    classifier_result_dict = defaultdict(list)
    for modelname in os.listdir(modelspath):
        if not any(modeltype in modelname for modeltype in ['TransE', 'DistMult']):
            continue
        if not os.path.isdir(join(modelspath, modelname)):
            continue

        print('#'*80+f'\nModelname: {modelname}\tmodelspath={modelspath}')

        model_params_dict = wot.utils.get_model_params(modelname, modelspath=modelspath)

        model = wot.utils.load_model(join(modelspath, modelname), which='best_')
        if cuda.is_available():
            model.cuda()

        # Binary classification features
        for sampling in ['Bernoulli', 'Positional']:
            bc_results = binary_classifiers_evaluation(modelname, train_kg,
                                                      modelspath=modelspath,
                                          whichmodel='best_', sampling=sampling)
            update_results_dict(classifier_result_dict, **model_params_dict, **bc_results,
                                evaluation_sampler=sampling)

        # Link Prediction
        lp_results = link_prediction_evaluation(model, test_kg, b_size=1, k=10)
        update_results_dict(lp_result_dict, **model_params_dict, **lp_results)


        # Triplet Classification
        #try:
        #  tc_results = triplet_classification_evaluation(model, tr_kg, val_kg, test_kg, b_size=1)
        #except Exception as e:
        #  print(e)
        #  pass
        #update_results_dict(tc_result_dict, **model_params_dict, **tc_results)


    return lp_result_dict, tc_result_dict, classifier_result_dict

def update_results_dict(results_dict, **kwargs):
    for k, v in kwargs.items():
        results_dict[k].append(v)
    return results_dict

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
