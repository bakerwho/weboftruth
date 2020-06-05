import pandas as pd
import pickle
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

import sys
import argparse
parser = argparse.ArgumentParser()

## pathnames
# args.path = "~/weboftruth"
# args.path = "/project2/jevans/aabir/weboftruth/"
# args.path = "/Users/aabir/Documents/research/weboftruth"
################################################################

#-p wot_path -e epochs -m model_type -small False
parser.add_argument("-p", "--path", dest="path",
                        default="/project2/jevans/aabir/weboftruth/",
                        help="path to weboftruth")
parser.add_argument("-e", "--epochs", dest="epochs",
                        default=100,
                        help="no. of training epochs", type=int)
parser.add_argument("-m", "--model", dest='model_type',
                        default='TransE',
                        help="model type")
parser.add_argument("-s", "--small", dest='small', default=False,
                        help="train small dataset", type=bool)

args = parser.parse_args()

svo_data_path = join(args.path, 'data/SVO-tensor-dataset')

models_path = join(args.path, 'models')

os.makedirs(models_path, exist_ok=True)

for f in os.listdir(svo_data_path):
    if 'train' in f: tr_fp = f
    if 'valid' in f: val_fp = f
    if 'test' in f: test_fp = f

def read_data(tr_fp, val_fp, test_fp):
    tr_df = pd.read_csv(join(svo_data_path, tr_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    val_df = pd.read_csv(join(svo_data_path, val_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    test_df = pd.read_csv(join(svo_data_path, test_fp),
                       sep='\t', header=None, names=['from', 'rel', 'to'])
    return tr_df, val_df, test_df

class CustomTransModel():
    def __init__(self, kg, model_type, **kwargs):
        self.kg = kg
        self.model_type = model_type
        self.diss_type = kwargs.pop('diss_type', 'L2')
        if model_type in ['TransR', 'TransD', 'TorusE']:
            self.ent_emb_dim = kwargs.pop('ent_emb_dim', 250)
            self.rel_emb_dim = kwargs.pop('rel_emb_dim', 250)
            self.model = getattr(torchkge.models, model_type + 'Model'
                                    )(ent_emb_dim=self.ent_emb_dim,
                                        rel_emb_dim=self.rel_emb_dim,
                                        n_entities=self.kg.n_ent,
                                        n_relations=self.kg.n_rel)
        else:
            self.emb_dim = kwargs.pop('emb_dim', 250)
            self.model = getattr(torchkge.models, f'{model_type}Model'
                                )(emb_dim=self.emb_dim, n_entities=kg.n_ent,
                                    n_relations=kg.n_rel,
                                    dissimilarity_type=self.diss_type)
        i = len([d for d in os.listdir(models_path
                                ) if os.path.isdir(join(models_path, d))
                                    and 'trial_' in d])
        self.model_path = join(models_path, f'trial_{str(i+1).zfill(2)}')
        os.makedirs(self.model_path, exist_ok=True)
        ## Hyperparameters
        self.lr = kwargs.pop('lr', 0.0004)
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.b_size = kwargs.pop('b_size', 32)

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
        epoch_loss = running_loss/i
        self.tr_losses.append(epoch_loss)
        return epoch_loss

    def validate(self, val_kg):
        losses = []
        for batch in DataLoader(val_kg, batch_size=self.b_size):
            h, t, r = batch
            n_h, n_t = self.sampler.corrupt_batch(h, t, r)
            pos, neg = self.model(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            losses.append(loss.item())
        return np.mean(losses)

    def train_model(self, n_epochs, val_kg):
        self.model.normalize_parameters()
        epochs = tqdm(range(n_epochs), unit='epoch')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            if (epoch+1%100)==0 or epoch==0:
                torch.save(self.model.state_dict(), join(self.model_path,
                                                    'transe_model.pt'))
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.best_epoch = self.epochs
                    torch.save(self.model.state_dict(), join(self.model_path,
                                f'best_{self.model_type}_model.pt'))
            epochs.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(self.epochs + 1, mean_epoch_loss))

class CustomBilinearModel(torchkge.models.interfaces.BilinearModel):
    def __init__(self, kg, model_type, **kwargs):
        self.kg = kg
        self.emb_dim = kwargs.pop('emb_dim', 250)
        self.model_type = model_type
        super().__init__(self.emb_dim, self.kg.n_ent, self.kg.n_rel)
        self.model = getattr(torchkge.models, self.model_type + 'Model')(self.emb_dim, n_entities = self.kg.n_ent,
                            n_relations = self.kg.n_rel)

        ## Hyperparameters
        self.lr = kwargs.pop('lr', 0.0004)
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.b_size = kwargs.pop('b_size', 32)

        try:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            self.dataloader = DataLoader(self.kg, batch_size=self.b_size)

        ## Logger
        self.epochs=0
        self.tr_losses=[]
        self.best_epoch=-1
        self.val_losses=[]

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
        self.normalize_parameters()
        self.epochs += 1
        epoch_loss = running_loss/i
        self.tr_losses.append(epoch_loss)
        return epoch_loss

    def validate(self, val_kg):
        losses = []
        for batch in DataLoader(val_kg, batch_size=self.b_size):
            h, t, r = batch
            n_h, n_t = self.sampler.corrupt_batch(h, t, r)
            pos, neg = self.model(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            losses.append(loss.item())
        return np.mean(losses)

    def train_model(self, n_epochs, val_kg):
        self.model.normalize_parameters()
        epochs = tqdm(range(n_epochs), unit='epoch')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            print(f'Epoch {self.epochs} | Train loss: {mean_epoch_loss}')
            if (epoch+1%100)==0 or epoch==0:
                torch.save(self.state_dict(), join(self.model_path,
                                    f'epoch_{self.epochs}_{self.model_type}_model.pt'))
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.best_epoch = epoch
                    torch.save(self.state_dict(), join(self.model_path,
                                f'best_{self.model_type}_model.pt'))
                print(f'\tEpoch {self.epochs} | Validation loss: {val_loss}')
                self.val_losses.append(val_loss)
                self.val_epochs.append(self.epochs)
        print(f"best epoch: {self.best_epoch}")


if __name__ == '__main__':
    print(f"Path: {args.path}\nModel Type: {args.model_type}")
    print(f"Epochs: {args.epochs}\nSmall: {args.small}")
    tr_df, val_df, test_df = read_data(tr_fp, val_fp, test_fp)
    sizes = [df.shape[0] for df in (tr_df, val_df, test_df)]
    full_df = pd.concat([tr_df, val_df, test_df])
    full_kg = torchkge.data_structures.KnowledgeGraph(full_df)
    tr_kg, val_kg, test_kg = full_kg.split_kg(sizes=sizes)
    try:
        if args.small:
            mod = CustomTransModel(test_kg, model_type = args.model_type)
        else:
            mod = CustomTransModel(tr_kg, model_type = args.model_type)
    except:
        mod = CustomBilinearModel(tr_kg, model_type = arg.model_type)
    mod.set_sampler(samplerClass=BernoulliNegativeSampler, kg=tr_kg)
    mod.set_optimizer(optClass=Adam)
    mod.set_loss(lossClass=MarginLoss, margin=0.5)
    # Move everything to CUDA if available
    if cuda.is_available():
        print("Using cuda.")
        cuda.empty_cache()
        mod.model.cuda()
        mod.loss_fn.cuda()
    mod.train_model(args.epochs, val_kg)
