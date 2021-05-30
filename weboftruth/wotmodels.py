import pandas as pd
import os
from os.path import join
import numpy as np

import torch
from torch import cuda
from torch.optim import Adam

import torchkge
from torchkge.models import Model, TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader

from tqdm.autonotebook import tqdm
from datetime import datetime
from tabulate import tabulate

import weboftruth as wot
#from weboftruth.constants import models_path, svo_paths, svo_data_path
from weboftruth import utils
from weboftruth.corrupt import corrupt_kg

import sys
import argparse
parser = argparse.ArgumentParser()

torch.manual_seed(0)

## pathnames
# args.path = "~/weboftruth"
# args.path = "/project2/jevans/aabir/weboftruth/"
# args.path = "/home-nfs/tenzorok/weboftruth"
################################################################

#-p wot_path -e epochs -m model_type -small False
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
parser.add_argument("-s", "--small", dest='small', default=False,
                        help="train small dataset", type=bool)
parser.add_argument("-ts", "--truthshare", dest="ts", default=100,
                        help="truth share of dataset", type=int)
parser.add_argument("-ve", "--valevery", dest="ve", default=10,
                        help="validate every X epochs", type=int)
#parser.add_argument("-name", dest="name", default=100,
#                        help="name for model saving", type=int)

args, unknown = parser.parse_known_args()

#svo_data_path = join(args.path, 'data', 'SVO-tensor-dataset')
#svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 90, 80, 50]}

args.modelpath = args.modelpath

try:
    os.makedirs(args.modelpath, exist_ok=True)
except:
    print("Warning: models folder may not exist")

class CustomTransModel():
    """
    Class containing Translation Embedding Models (Torch-KGE) and the
    Knowledge Graph to be trained on

    The Model and KG are both stored in the folder `self.model_folder`

    Use load_WOTmodel(model_folder) after initialization to load from disc
    """
    def __init__(self, trainkg, traints, model_type, bilinear=False, **kwargs):
        # add train_ts/truth_share parameter
        self.trainkg = trainkg
        self.traints = traints
        self.model_type = model_type
        self.diss_type = kwargs.pop('diss_type', 'L2')
        self.dataset_name = kwargs.pop('dataset_name', None)
        if self.dataset_name:
            self.logline(f'Dataset: {self.dataset_name}')
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
            if self.model_type is 'TransE':
                self.model = getattr(torchkge.models, f'{model_type}Model'
                                )(emb_dim=self.emb_dim,
                                    n_entities=self.trainkg.n_ent,
                                    n_relations=self.trainkg.n_rel,
                                    dissimilarity_type=self.diss_type)
            elif bilinear:
                self.model = getattr(torchkge.models.bilinear, self.model_type + 'Model'
                                    )(emb_dim=self.emb_dim,
                                        n_entities = self.trainkg.n_ent,
                                        n_relations = self.trainkg.n_rel)
            else:
                self.model = getattr(torchkge.models, f'{model_type}Model'
                                )(emb_dim=self.emb_dim,
                                    n_entities=self.trainkg.n_ent,
                                    n_relations=self.trainkg.n_rel)
        self.n_entities = self.trainkg.n_ent
        self.n_relations = self.trainkg.n_rel

        self.set_model_path(kwargs.pop('model_path', None))
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
        print(f'Creating {self.model_type} in folder: {self.model_path}')

        # Legacy code
        # super(CustomTransModel, self).__init__(self.emb_dim, self.trainkg.n_ent, self.trainkg.n_rel,
        #                     dissimilarity_type=self.diss_type)


        try:
            self.dataloader = DataLoader(self.trainkg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            self.dataloader = DataLoader(self.trainkg, batch_size=self.b_size)

        ## Logger
        self.epochs=0
        self.tr_losses=[]
        self.best_epoch=-1
        self.val_losses=[]
        self.val_epochs=[]

    def set_model_path(self, folder_name=None):
        if folder_name is not None:
            self.model_path = folder_name
            return
        all_is = [int(d.split('_')[1]) for d in os.listdir(wot.args.modelpath)
                        #all items in model path
                        if os.path.isdir(join(wot.args.modelpath, d)
                        # that are directories
                        ) and f'{self.model_type}_' in d]
                        #and are of type self.model_type
        i = [x for x in range(1, len(all_is)+2) if x not in all_is][0]
        ds = self.dataset_name
        ds = ds+'_' if ds else ''
        self.model_path = join(wot.args.modelpath, f'{self.model_type}_{ds}{str(i).zfill(2)}')
        print(f" saving model to {self.model_path}")
        os.makedirs(self.model_path, exist_ok=True)

    def logline(self, line):
        with open(self.logfile, 'a+') as f:
            f.write(line)
            f.write('\n')

    def set_optimizer(self, optClass=Adam, **kwargs):
        self.optimizer = optClass(self.model.parameters(), lr=self.lr,
                                    **kwargs)

    def set_sampler(self, samplerClass=BernoulliNegativeSampler, **kwargs):
        self.sampler = samplerClass(**kwargs)

    def set_loss(self, lossClass=MarginLoss, **kwargs):
        self.loss_fn = lossClass(**kwargs)
        try:
            self.loss_fn.cuda()
        except:
            pass

    def one_epoch(self):
        running_loss = 0.0
        for i, batch in enumerate(self.dataloader):
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

    def validate(self, val_kg, istest=False, verbose=False):
        losses = []
        try:
            dataloader = DataLoader(val_kg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            dataloader = DataLoader(val_kg, batch_size=self.b_size)

        for batch in dataloader:
            h, t, r = batch
            n_h, n_t = self.sampler.corrupt_batch(h, t, r)
            pos, neg = self.model(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            losses.append(loss.item())
            if istest:
                self.logline(f'\t\tTest loss: {np.mean(losses)}')
        if verbose:
            print(f'\t\tTest loss after epoch {self.epochs}: {np.mean(losses)}')
        return np.mean(losses)

    def train_model(self, n_epochs, val_kg):
        epochs = tqdm(range(n_epochs), unit='epoch')
        if self.epochs == 0:
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' UTC'
            self.logline(f'Training started at {dt}\n')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            self.logline(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if ((epoch+1)%args.ve)==0 or epoch==0:
                self.save_model()
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.save_model(best=True)
                self.logline(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                self.val_losses.append(val_loss)
                self.val_epochs.append(self.epochs)
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
            modelname = 'best_'+modelname
        modelpath = join(self.model_path, modelname+'.pt')
        print(f' saving {self.model_type} to {modelpath}')
        torch.save(self.model.state_dict(), modelpath)

    def save_kg(self, kg, addtxt=''):
        # save knowledge Graph
        if kg is None:
            kg = self.trainkg
        df = utils.kg_to_df(kg)
        kgdfname = f'{addtxt}_{self.model_type}_kg.csv'
        kgdfpath = join(self.model_path, kgdfname)
        if not os.path.exists(kgdfpath):
            df.to_csv(kgdfpath, index=False)
        print(f' saving trainkg to {kgdfpath}')

    def load_WOTmodel(self, model_path=None, which='best_'):
        if model_path is None:
            model_path = self.model_path
        self.model = utils.load_model(model_path, which)
        self.kg = utils.load_kg(model_path)

def modelslist(module):
    return [x for x in dir(module) if 'model' in x.lower()]

if __name__ == '__main__':
    print(f"Datapath: {args.datapath}\nModelpath: {args.modelpath}")
    print(f"Dataset: {args.dataset}\n")
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}\nSmall: {args.small}")
    print(f"Truth share: {args.ts}")
    #tr_fn, val_fn, test_fn = wot.utils.get_svo_file_names(args.ts)
    tr_fn, val_fn, test_fn = wot.utils.get_github_filenames(args.datapath,
                                args.dataset)
    #print(tr_fn)
    dfs = wot.utils.read_data(tr_fn, val_fn, test_fn,
                                join(args.datapath, args.dataset))
    dfs = [df.drop('true_positive', axis=1
                ) if 'true_positive' in df.columns else df
                for df in dfs ]
    tr_kg, val_kg, test_kg = (wot.utils.df_to_kg(df) for df in dfs)
    sizes = [df.shape[0] for df in dfs]
    #full_df = pd.concat([dfs[0], dfs[1], dfs[2]])
    #full_kg = wot.utils.df_to_kg(full_df)
    #tr_kg, val_kg, test_kg = full_kg.split_kg(sizes=sizes)
    if args.model_type+'Model' in modelslist(torchkge.models.translation):
        if args.small:
            mod = CustomTransModel(trainkg=test_kg, traints=args.ts,
                                    model_type=args.model_type)
        else:
            mod = CustomTransModel(trainkg=tr_kg, traints=args.ts,
                                        model_type=args.model_type)
    elif args.model_type+'Model' in modelslist(torchkge.models.bilinear):
        if args.small:
            mod = CustomBilinearModel(trainkg=tr_kg, traints=args.ts,
                                model_type=args.model_type)
        else:
            mod = CustomBilinearModel(trainkg=tr_kg, traints=args.ts,
                                    model_type=args.model_type)
    mod.set_sampler(samplerClass=BernoulliNegativeSampler, kg=tr_kg)
    mod.set_optimizer(optClass=Adam)
    mod.set_loss(lossClass=MarginLoss, margin=0.5)
    # Move everything to CUDA if available
    if cuda.is_available():
        print("Using cuda.")
        cuda.empty_cache()
        cuda.init()
        mod.model.cuda()
        mod.loss_fn.cuda()
    mod.set_model_path(args.modelpath)
    mod.train_model(args.epochs, tr_kg)
    #mod.save_kg(test_kg, 'test')
    print('\nTest set performance:')
    mod.validate(test_kg, istest=True, verbose=True)
    #mod.save_kg(val_kg, 'val')
    print('\nValidation set performance:')
    mod.validate(val_kg, istest=True, verbose=True)
