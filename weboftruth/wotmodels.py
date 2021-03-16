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
from weboftruth.constants import models_path, svo_paths, svo_data_path
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
parser.add_argument("-p", "--path", dest="path",
                        #default="/home/ubuntu/weboftruth/",
                        #default="/project2/jevans/aabir/weboftruth/",
                        default="/home-nfs/tenzorok/weboftruth",
                        help="path to weboftruth")
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

svo_data_path = join(args.path, 'data', 'SVO-tensor-dataset')
svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 90, 80, 50]}

models_path = join(args.path, 'models')

try:
    os.makedirs(models_path, exist_ok=True)
except:
    print("Warning: models folder may not exist")

class CustomTransModel():
    """
    Class containing Translation Embedding Models (Torch-KGE) and the
    Knowledge Graph to be trained on

    The Model and KG are both stored in the folder `self.model_folder`

    Use load_model(model_folder) after initialization to load from disc
    """
    def __init__(self, kg, model_type, **kwargs):
        self.kg = kg
        self.model_type = model_type
        self.diss_type = kwargs.pop('diss_type', 'L2')
        if self.model_type in ['TransR', 'TransD', 'TorusE']:
            self.ent_emb_dim = kwargs.pop('ent_emb_dim', args.emb_dim)
            self.rel_emb_dim = kwargs.pop('rel_emb_dim', args.emb_dim)
            self.model = getattr(torchkge.models.translation, model_type + 'Model'
                                    )(ent_emb_dim=self.ent_emb_dim,
                                        rel_emb_dim=self.rel_emb_dim,
                                        n_entities=self.kg.n_ent,
                                        n_relations=self.kg.n_rel)
        else:
            self.emb_dim = kwargs.pop('emb_dim', args.emb_dim)
            if self.model_type is 'TransE':
                self.model = getattr(torchkge.models, f'{model_type}Model'
                                )(emb_dim=self.emb_dim, n_entities=kg.n_ent,
                                    n_relations=kg.n_rel,
                                    dissimilarity_type=self.diss_type)
            else:
                self.model = getattr(torchkge.models, f'{model_type}Model'
                                )(emb_dim=self.emb_dim, n_entities=kg.n_ent,
                                    n_relations=kg.n_rel)
        self.n_entities = kg.n_ent
        self.n_relations = kg.n_rel
        print('!!', kg.n_ent, kg.n_rel)

        self.setup_model_folder()

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
        # super(CustomTransModel, self).__init__(self.emb_dim, kg.n_ent, kg.n_rel,
        #                     dissimilarity_type=self.diss_type)


        try:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size)

        ## Logger
        self.epochs=0
        self.tr_losses=[]
        self.best_epoch=-1
        self.val_losses=[]
        self.val_epochs=[]

    def setup_model_folder(self):
        all_is = [int(d.split('_')[1]) for d in os.listdir(wot.models_path
                        ) if os.path.isdir(join(wot.models_path, d)
                        ) and f'{self.model_type}_' in d]
        i = [x for x in range(1, len(all_is)+2) if x not in all_is][0]
        self.model_path = join(models_path, f'{self.model_type}_{str(i+1).zfill(2)}')
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

    def validate(self, val_kg, istest=False):
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
                self.logline('\t\tTest loss: {np.mean(losses)}')
        return np.mean(losses)

    def train_model(self, n_epochs, val_kg, ts=''):
        epochs = tqdm(range(n_epochs), unit='epoch')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            self.logline(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if ((epoch+1)%args.ve)==0 or epoch==0:
                self.save_model(ts=ts)
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.save_model(best=True, ts=ts)
                self.logline(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                self.val_losses.append(val_loss)
                self.val_epochs.append(self.epochs)
        self.logline(f"\nbest epoch: {self.best_epoch}\n")
        self.model.normalize_parameters()

    def save_model(self, best=False, ts=''):
        # files written to self.model_path
        # save torch model
        ts = f"_ts={ts}" if not ts else ts
        modelname = f'e={self.epochs}{ts}_{self.model_type}_model'
        if best:
            self.best_epoch = self.epochs
            modelname = 'best_'+modelname
        modelpath = join(self.model_path, modelname+'.pt')
        print(f' saving f{self.model_type} to {modelpath}')
        torch.save(self.model.state_dict(), modelpath)

        # save knowledge Graph
        df = utils.kg_to_df(kg)
        kgdfname = f'{self.model_type}_kg.csv'
        kgdfpath = join(self.model_path, kgdfname)
        if not os.exists(kgdfpath):
            df.to_csv(kgdfpath, index=False)

    def load_model(self, model_path, which='best_'):
        self.model = utils.load_model(model_path, which)
        self.kg = utils.load_kg(model_path)


class CustomBilinearModel():
    def __init__(self, kg, model_type, **kwargs):
        self.kg = kg
        self.emb_dim = kwargs.pop('emb_dim', 250)
        self.model_type = model_type
        self.model = getattr(torchkge.models.bilinear, self.model_type + 'Model'
                            )(emb_dim=self.emb_dim,
                                n_entities = self.kg.n_ent,
                                n_relations = self.kg.n_rel)
        all_is = [int(d.split('_')[1]) for d in os.listdir(models_path
                                ) if os.path.isdir(join(models_path, d))
                                    and f'{self.model_type}_' in d]
        i = [x for x in range(1, len(all_is)+2) if x not in all_is][0]
        self.model_path = join(models_path, f'{self.model_type}_{str(i+1).zfill(2)}')
        os.makedirs(self.model_path, exist_ok=True)
        self.logfile = join(self.model_path, 'log.txt')

        ## Hyperparameters
        self.lr = kwargs.pop('lr', 0.0004)
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.b_size = kwargs.pop('b_size', 32)
        self.logline(tabulate([(k,v) for k, v in vars(self).items()],
                                    headers=['variable', 'value']))

        try:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size)

        ## Logger
        self.epochs=0
        self.tr_losses=[]
        self.best_epoch=-1
        self.val_losses=[]
        self.val_epochs=[]

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
            pos, neg = self.model(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.model.normalize_parameters()
        self.epochs += 1
        epoch_loss = running_loss/i
        self.tr_losses.append(epoch_loss)
        return epoch_loss

    def validate(self, val_kg, istest=False):
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
            self.logline('\t\tTest loss: {np.mean(losses)}')
        return np.mean(losses)

    def train_model(self, n_epochs, val_kg, ts=''):
        epochs = tqdm(range(n_epochs), unit='epoch')
        for epoch in epochs:
            if self.epochs == 0:
                dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' UTC'
                self.logline(f'Training started at {dt}\n')
            mean_epoch_loss = self.one_epoch()
            self.logline(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if ((epoch+1)%args.ve)==0 or epoch==0:
                self.save_model(ts=ts)
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.save_model(best=True, ts=ts)
                self.logline(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                self.val_losses.append(val_loss)
                self.val_epochs.append(self.epochs)
        self.logline(f"\nbest epoch: {self.best_epoch}\n")
        self.model.normalize_parameters()

    def save_model(self, best=False, ts=''):
        # files written to self.model_path
        # save torch model
        ts = f"_ts={ts}" if not ts else ts
        modelname = f'e={self.epochs}{ts}_{self.model_type}_model'
        if best:
            self.best_epoch = self.epochs
            modelname = 'best_'+modelname
        modelpath = join(self.model_path, modelname+'.pt')
        print(f' saving f{self.model_type} to {modelpath}')
        torch.save(self.model.state_dict(), modelpath)

        # save knowledge Graph
        df = utils.kg_to_df(kg)
        kgdfname = f'{self.model_type}_kg.csv'
        kgdfpath = join(self.model_path, kgdfname)
        if not os.exists(kgdfpath):
            df.to_csv(kgdfpath)

    def load_model(self, model_path, which='best_'):
        self.model = utils.load_model(model_path, which)
        self.kg = utils.load_kg(model_path)

def modelslist(module):
    return [x for x in dir(module) if 'model' in x.lower()]



if __name__ == '__main__':
    print(f"Path: {args.path}\nModel Type: {args.model_type}")
    print(f"Epochs: {args.epochs}\nSmall: {args.small}")
    print(f"Truth share: {args.ts}")
    tr_fn, val_fn, test_fn = wot.utils.get_file_names(args.ts)
    print(tr_fn)
    dfs = wot.utils.read_data(tr_fn, val_fn, test_fn,
                                svo_paths[args.ts])
    dfs = [df.drop('true_positive', axis=1
                ) if 'true_positive' in df.columns else df
                for df in dfs ]
    #tr_kg, val_kg, test_kg = (wot.utils.df_to_kg(df) for df in dfs)
    sizes = [df.shape[0] for df in dfs]
    full_df = pd.concat([dfs[0], dfs[1], dfs[2]])
    full_kg = wot.utils.df_to_kg(full_df)
    tr_kg, val_kg, test_kg = full_kg.split_kg(sizes=sizes)
    print(tr_kg.n_rel)
    if args.model_type+'Model' in modelslist(torchkge.models.translation):
        if args.small:
            mod = CustomTransModel(test_kg, model_type=args.model_type,
                                        ts=args.ts)
        else:
            mod = CustomTransModel(tr_kg, model_type=args.model_type,
                                        ts=args.ts)
    elif args.model_type+'Model' in modelslist(torchkge.models.bilinear):
        if args.small:
            mod = CustomBilinearModel(tr_kg, model_type=args.model_type,
                                        ts=args.ts)
        else:
            mod = CustomBilinearModel(tr_kg, model_type=args.model_type,
                                        ts=args.ts)
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
    mod.train_model(args.epochs, val_kg)
    mod.validate(test_kg, istest=True)
