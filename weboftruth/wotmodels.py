from copy import deepcopy

from easydict import EasyDict as edict

import pandas as pd
import os
from os.path import join
import numpy as np

import torch
from torch import cuda
from torch.optim import Adam

import torchkge
from torchkge.models import Model, TransEModel
from torchkge.sampling import *
from torchkge.utils import MarginLoss, DataLoader
from torchkge.evaluation.triplet_classification import TripletClassificationEvaluator
from torchkge.evaluation.link_prediction import LinkPredictionEvaluator

from tqdm.autonotebook import tqdm
from datetime import datetime
from tabulate import tabulate

import weboftruth as wot
#from weboftruth.constants import models_path, svo_paths, svo_data_path
from weboftruth import utils

import sys
import argparse


torch.manual_seed(0)

## pathnames
# args.path = "~/weboftruth"
# args.path = "/project2/jevans/aabir/weboftruth/"
# args.path = "/home-nfs/tenzorok/weboftruth"
################################################################

USE_CUDA_DEFAULT = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dpath", dest="datapath",
                            help="path to data")
    parser.add_argument("-ds", "--dataset", dest="dataset",
                            help="dataset name", type=str)
    parser.add_argument("-mp", "--mpath", dest="modelpath",
                            help="path to models")
    parser.add_argument("-e", "--epochs", dest="epochs",
                            default=100,
                            help="no. of training epochs", type=int)
    parser.add_argument("-m", "--model", dest='model_type',
                            default='TransE',
                            help="model type")
    parser.add_argument("-lr", "--learningrate", dest='lr',
                                default=5e-5,
                                help="learning rate", type=float)
    parser.add_argument("-emb", "--embdim", dest='emb_dim',
                            default=250,
                            help="embedding dimension", type=int)
    parser.add_argument("-test", "--testrun", dest='is_test_run', default=False,
                            help="train on (smaller) test dataset",
                            action='store_true')
    parser.add_argument("-ts", "--truthshare", dest="ts", default=100,
                            help="truth share of dataset", type=int)
    parser.add_argument("-ve", "--valevery", dest="ve", default=10,
                            help="validate every X epochs", type=int)
    parser.add_argument("-shuffle", "--shuffle", dest="shuffle", default=False,
                            help="to shuffle data at datapath",
                            action='store_true')
    parser.add_argument("-filters", "--numfilters", dest="n_filters", default=3,
                            help="no. of convolutional filters", type=int)
    parser.add_argument("-trsampler", "--trainsampler", dest="train_sampler",
                            default='BernoulliNegativeSampler',
                            help="Traintime negative sampler", type=str)
    parser.add_argument("-corrsampler", "--corruptsampler",
                    dest="corruption_sampler", default='BernoulliNegativeSampler',
                    help="Negative sampler for corruption", type=str)
    parser.add_argument("-cuda", "--use_cuda", dest="use_cuda",
                    default=USE_CUDA_DEFAULT, help="To use cuda",
                    action='store_true')
    return parser

#svo_data_path = join(args.path, 'data', 'SVO-tensor-dataset')
#svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 90, 80, 50]}

args = edict({  "epochs":100, "model_type":'TransE', "lr":5e-5,
                "emb_dim":250, "is_test_run":False, "ve":10,
                "shuffle":False, "n_filters":3,
                "train_sampler":'BernoulliNegativeSampler',
                "corruption_sampler":'BernoulliNegativeSampler',
                "use_cuda": USE_CUDA_DEFAULT,
                "modelpath": '../../models',
                "is_test_run": False,
                "dataset": 'FB15K-237',
                "ts":100
                })

class CustomKGEModel():
    """
    Class containing Translation Embedding Models (Torch-KGE) and the
    Knowledge Graph to be trained on

    The Model and KG are both stored in the folder `self.model_folder`

    Use load_WOTmodel(model_folder) after initialization to load from disc
    """
    def __init__(self, trainkg, traints, model_type, **kwargs):
        # add train_ts/truth_share parameter
        self.trainkg = trainkg
        self.traints = traints
        self.model_type = model_type.replace("Model", '')
        self.dataset_name = kwargs.pop('dataset_name', None)

        if self.dataset_name:
            self.logline(f'Dataset: {self.dataset_name}')

        self.dataloader_cuda_flag = 'all' if (args.use_cuda and
                                            cuda.is_available()) else None

        if self.model_type in ['TransR', 'TransD', 'TorusE']:
            self.ent_emb_dim = kwargs.pop('ent_emb_dim', args.emb_dim)
            self.rel_emb_dim = kwargs.pop('rel_emb_dim', args.emb_dim)
            self.model = getattr(torchkge.models.translation, model_type + 'Model'
                                    )(ent_emb_dim=self.ent_emb_dim,
                                        rel_emb_dim=self.rel_emb_dim,
                                        n_entities=self.trainkg.n_ent,
                                        n_relations=self.trainkg.n_rel)
        else:
            self.emb_dim = kwargs.pop('emb_dim', args.emb_dim)
            if self.model_type == 'TransE':
                self.diss_type = kwargs.pop('diss_type', 'L2')
                self.model = getattr(torchkge.models, 'TransEModel'
                                )(emb_dim=self.emb_dim,
                                    n_entities=self.trainkg.n_ent,
                                    n_relations=self.trainkg.n_rel,
                                    dissimilarity_type=self.diss_type)
            elif any((self.model_type in x) for x in dir(
                                                    torchkge.models.bilinear)):
                self.model = getattr(torchkge.models.bilinear,
                                    self.model_type + 'Model'
                                    )(emb_dim=self.emb_dim,
                                        n_entities = self.trainkg.n_ent,
                                        n_relations = self.trainkg.n_rel)
            elif self.model_type == 'ConvKB':
                self.n_filters = kwargs.pop('n_filters', args.n_filters)
                self.model = getattr(torchkge.models.deep,
                                    'ConvKBModel'
                                    )(emb_dim=self.emb_dim,
                                        n_filters=self.n_filters,
                                        n_entities = self.trainkg.n_ent,
                                        n_relations = self.trainkg.n_rel)
            else:
                raise ValueError(f'Invalid model_type: {self.model_type}')
        self.n_entities = self.trainkg.n_ent
        self.n_relations = self.trainkg.n_rel

        self.create_model_path(kwargs.pop('global_model_path', args.modelpath))
        if args.shuffle:
            self.save_kg(self.trainkg, addtxt='train')

        vars_df =  pd.DataFrame.from_dict(vars(self), orient='index')
        vars_df.to_csv(join(self.model_path, f'{self.model_type}_modelinfo.txt'))

        self.logfile = join(self.model_path, 'log.txt')
        ## Hyperparameters
        self.lr = kwargs.pop('lr', args.lr)
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.b_size = kwargs.pop('b_size', 32)
        self.logline(tabulate([(k,v) for k, v in vars(self).items()],
                                    headers=['variable', 'value']))
        self.ent_vecs, self.rel_vecs = None, None

        # Legacy code
        # super(CustomKGEModel, self).__init__(self.emb_dim, self.trainkg.n_ent, self.trainkg.n_rel,
        #                     dissimilarity_type=self.diss_type)

        self.train_dataloader = DataLoader(self.trainkg, batch_size=self.b_size,
                use_cuda=self.dataloader_cuda_flag)

        ## Logger
        self.epochs=0
        self.tr_losses=[]
        self.best_epoch=-1
        self.val_losses=[]
        self.val_epochs=[]

    def create_model_path(self, folder_name):
        all_is = [int(d.split('_')[1]) for d in os.listdir(folder_name)
                        #all items in model path
                        if os.path.isdir(join(args.modelpath, d)
                        # that are directories
                        ) and f'{self.model_type}_' in d
                        and d.split('_')[1].isnumeric()]
                        #and are of type self.model_type
        i = [x for x in range(1, len(all_is)+2) if x not in all_is][0]
        ds = self.dataset_name
        ds = ds+'_' if ds else ''
        self.model_name = f'{self.model_type}_{ds}{str(i).zfill(2)}'
        self.model_path = join(folder_name, self.model_name)
        print(f"{self.model_type} self.model_path: {self.model_path}")
        os.makedirs(self.model_path, exist_ok=True)

    def logline(self, line):
        with open(self.logfile, 'a+') as f:
            f.write(line)
            f.write('\n')

    def set_optimizer(self, optClass=Adam, **kwargs):
        self.optimizer = optClass(self.model.parameters(), lr=self.lr,
                                    **kwargs)
        self.logline(f'Optimizer set: {self.optimizer}')

    def set_train_neg_sampler(self, samplerClass=BernoulliNegativeSampler, **kwargs):
        self.sampler = samplerClass(**kwargs)
        self.logline(f'Traintime sampler set: {self.sampler}')

    def set_loss(self, lossClass=MarginLoss, **kwargs):
        self.loss_fn = lossClass(**kwargs)
        if cuda.is_available() and args.use_cuda:
            self.loss_fn.cuda()
        self.logline(f'Loss function set: {self.loss_fn}')

    def one_epoch(self):
        self.model.train(True)
        running_loss = 0.0
        for i, batch in enumerate(self.train_dataloader):
            h, t, r = batch
            n_h, n_t = self.sampler.corrupt_batch(h, t, r)
            self.optimizer.zero_grad()
            pos, neg = self.model.forward(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.model.normalize_parameters()
        self.epochs += 1
        epoch_loss = running_loss/(i+1)
        self.tr_losses.append(epoch_loss)
        return epoch_loss

    def eval_base(self, val_kg, istest=False, verbose=False):
        self.model.train(False)
        losses = []

        eval_dataloader = DataLoader(self.trainkg, batch_size=self.b_size,
                use_cuda=self.dataloader_cuda_flag)

        with torch.no_grad():
            for batch in eval_dataloader:
                h, t, r = batch
                n_h, n_t = self.sampler.corrupt_batch(h, t, r)
                pos, neg = self.model(h, t, n_h, n_t, r)
                loss = self.loss_fn(pos, neg)
                losses.append(loss.item())
        self.logline('Base Evaluator results:')
        if istest:
            self.logline(f'\t\tTest loss: {np.mean(losses)}')
        if verbose:
            print(f'\t\tTest loss after epoch {self.epochs}: {np.mean(losses)}')
        return np.mean(losses)

    def eval_link_predict(self, kg_test):
        self.model.train(False)
        with torch.no_grad():
            evaluator = LinkPredictionEvaluator(self.model, kg_test)
            evaluator.evaluate(b_size=32)
        with utils.Capturing() as out:
            evaluator.print_results()
        out = '\n'.join(out)
        self.logline('LinkPredictionEvaluator results:')
        self.logline(out)
        del out
        return evaluator

    def eval_triplet_predict(self, kg_val, kg_test):
        self.model.train(False)
        with torch.no_grad():
            evaluator = TripletClassificationEvaluator(self.model, kg_val, kg_test)
            evaluator.evaluate(b_size=32)
        self.logline('TripletClassificationEvaluator results:')
        self.logline(f"Triplet classification accuracy with"\
        f"{evaluator.sampler}: {evaluator.accuracy(b_size=16)}\n"\
        f"\t(threshold = {evaluator.thresholds})")
        return evaluator

    def train_model(self, n_epochs, val_kg, early_stopping=True, verbose=False):
        self.model.train(True)
        epochs = tqdm(range(n_epochs), unit='epoch')
        if self.epochs == 0:
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' UTC'
            self.logline(f'Training started at {dt}\n')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            self.logline(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if verbose:
                print(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if ((epoch+1)%args.ve)==0 or epoch==0:
                self.save_model()
                val_loss = self.eval_base(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    if self.epochs>1:
                        self.save_model(best=True)
                self.logline(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                self.val_losses.append(val_loss)
                self.val_epochs.append(self.epochs)
                if verbose:
                    print(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                if early_stopping and epoch>n_epochs//2:
                    min_val_loss = min(self.val_losses)
                    if (0 < (min_val_loss - val_loss)/min_val_loss < 0.01):
                        print(f'Stopping early at epoch = {epoch} '\
                        'as validation loss decreases by less than 1%')
                        break
        self.logline(f"\nbest epoch: {self.best_epoch}\n")
        self.model.normalize_parameters()

    def save_model(self, best=False):
        # files written to self.model_path
        # save torch model
        modelname = f'ts={self.traints}_{self.model_type}_model'
        if not best:
            modelname = f'e={self.epochs}'+modelname
        else:
            self.best_epoch = self.epochs
            modelname = f'best_e={self.epochs}'+modelname
        modelpath = join(self.model_path, modelname+'.pt')
        print(f' saving {self.model_type} to {modelpath}')
        torch.save(self.model.state_dict(), modelpath)

    def save_kg(self, kgdf, addtxt=''):
        # save knowledge Graph
        if kgdf is None:
            kg, addtxt = self.trainkg, 'train'
            df = utils.kg_to_df(kg)
        if isinstance(kgdf, pd.DataFrame):
            df = kgdf
        elif isinstance(kgdf, torchkge.KnowledgeGraph):
            df = utils.kg_to_df(kgdf)
        else:
            raise TypeError("Invalid argument kgdf must be"\
                            " DataFrame or KnowledgeGraph")
        kgdfname = f'{addtxt}_{self.model_type}_kg.csv'
        kgdfpath = join(self.model_path, kgdfname)
        if not os.path.exists(kgdfpath):
            df.to_csv(kgdfpath, index=False)
        print(f' saving {addtxt} kg to {kgdfpath}')

    def load_WOTmodel(self, model_path=None, which='best_'):
        if model_path is None:
            model_path = self.model_path
        self.model = utils.load_model(model_path, which)
        self.kg = utils.load_kg(model_path)

def modelslist(module):
    return [x for x in dir(module) if 'model' in x.lower()]

if __name__ == '__main__':
    parser = get_parser()

    args, unknown = parser.parse_known_args()

    try:
        os.makedirs(args.modelpath, exist_ok=True)
    except:
        print("Warning: models folder may not exist")

    print(f"Datapath: {args.datapath}\nGlobal modelpath: {args.modelpath}")
    print(f"Dataset: {args.dataset}\n")
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}\nRun on test dataset: {args.is_test_run}")
    print(f"Truth share: {args.ts}\nEmbedding dimension: {args.emb_dim}")
    print(f"Train sampler: {args.train_sampler}")
    print(f"Corruption sampler: {args.corruption_sampler}")

    print(f"Using cuda: {args.use_cuda}")

    # Load data
    #tr_fn, val_fn, test_fn = wot.utils.get_svo_file_names(args.ts)
    as_id = 'FB15K' not in args.dataset
    tr_fn, val_fn, test_fn = wot.utils.get_simonepri_filenames(args.datapath,
                                                            args.dataset,
                                                            id=as_id)
    #explode = 'FB15K' not in args.dataset
    explode = False
    dfs = wot.utils.read_data(tr_fn, val_fn, test_fn,
                             join(args.datapath, args.dataset),
                    explode_rels=explode, rel_sep='/',
                    colnames=['from', 'rel', 'to'])

    # optionally shuffle dataset
    if args.shuffle:
        print('Warning: shuffling train/val/test datasets')
        tr_kg, val_kg, test_kg = wot.utils.reshuffle_trte_split(dfs)
        for txt, kg in zip(['train_shuffled', 'val_shuffled', 'test_shuffled'],
                           [tr_kg, val_kg, test_kg]):
            mod.save_kg(kg, txt)
    else:
        tr_kg, val_kg, test_kg = (wot.utils.df_to_kg(df) for df in dfs)


    # Initialize model

    model_args = {'trainkg': tr_kg, 'traints': args.ts,
                 'model_type':args.model_type, 'emb_dim': args.emb_dim,
                 'dataset':args.dataset}

    mod = CustomKGEModel(**model_args)
    sampler = getattr(torchkge.sampling, args.train_sampler)
    mod.set_train_neg_sampler(samplerClass=sampler, kg=tr_kg)
    mod.set_optimizer(optClass=Adam)
    mod.set_loss(lossClass=MarginLoss, margin=0.1)

    print(f'Model Name: {mod.model_name}\tModel Path: {mod.model_path}')

    # corrupt training KG if required
    if args.ts != 100:
        tr_kg_pure = deepcopy(tr_kg)
        shuffletxt = '_shuffle' if args.shuffle else ''
        sampler2 = sampler = getattr(torchkge.sampling, args.corruption_sampler)
        tr_kg, _ = wot.corrupt.corrupt_kg(tr_kg, save_folder=mod.model_path,
                        sampler=sampler2,
                        true_share=args.ts/100, use_cuda=False,
                        prefilename=f'ts={args.ts}_corrupt_{tr_fn}{shuffletxt}')
        mod.save_kg(tr_kg, f'ts={args.ts}_corrupt_{tr_fn}{shuffletxt}')

    # Move everything to CUDA if available
    if cuda.is_available() and args.use_cuda:
        print("Using cuda.")
        cuda.empty_cache()
        cuda.init()
        mod.model.cuda()
        mod.loss_fn.cuda()

    #mod.create_model_path(args.modelpath)
    mod.train_model(args.epochs, tr_kg)
    print('\nTest set base evaluator:')
    mod.eval_base(test_kg, istest=True, verbose=True)
    print('\nValidation set base evaluator:')
    mod.eval_base(val_kg, istest=True, verbose=True)
    print('Torchkge evaluators:')
    mod.eval_link_predict(test_kg)
    mod.eval_triplet_predict(val_kg, test_kg)
