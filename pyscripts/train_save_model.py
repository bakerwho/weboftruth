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

## pathnames-adarshm
wot_path = './'
svo_data_path = join(wot_path, 'data/SVO-tensor-dataset')

## pathnames-aabir

# fix this
#wot_path = "~/weboftruth"
##wot_path = "/project2/jevans/aabir/weboftruth/"
#wot_path = "/Users/aabir/Documents/research/weboftruth"
##svo_data_path = join(wot_path, 'SVO-tensor-dataset')
################################################################

model_path = join(wot_path, 'models')

os.makedirs(model_path, exist_ok=True)

files = os.listdir(svo_data_path)
for f in files:
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

class CustomTransModel(torchkge.models.interfaces.TranslationModel):
    def __init__(self, kg, model_type, **kwargs):
        self.kg = kg
        self.model_type = model_type
        self.diss_type = kwargs.pop('diss_type', 'L2')
        super().__init__(self.kg.n_ent, self.kg.n_rel, self.diss_type)

        if model_type in ['TransR', 'TransD', 'TorusE']:
            self.ent_emb_dim = kwargs.pop('ent_emb_dim', 250)
            self.rel_emb_dim = kwargs.pop('rel_emb_dim', 250)
            self.model = getattr(torchkge.models, model_type + 'Model')(self.ent_emb_dim,
                                    self.rel_emb_dim, n_entities = self.kg.n_ent, n_relations = self.kg.n_rel)
        else:
            self.emb_dim = kwargs.pop('emb_dim', 250)
            self.model = getattr(torchkge.models, model_type + 'Model')(self.emb_dim, n_entities = self.kg.n_ent,
                                    n_relations = self.kg.n_rel)

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
                torch.save(self.model.state_dict(), join(model_path,
                                                    'transe_model.pt'))
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), join(model_path,
                                'best_', self.model_type,'_model.pt'))
            epochs.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, mean_epoch_loss))

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
<<<<<<< HEAD
        self.normalize_parameters()
=======
        self.model.normalize_parameters()
>>>>>>> 12c0a976135ba658ad84c5b9fcf442122bab2980
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
<<<<<<< HEAD
                torch.save(model.state_dict(), join(model_path,
                                                    f'epoch_{self.epochs}_transe_model.pt'))
                val_loss = validate(val_kg)
                if not val_losses or val_loss < min(val_losses):
=======
                torch.save(self.model.state_dict(), join(model_path,
                                                    'transe_model.pt'))
                val_loss = self.validate(val_kg)
                if not self.val_losses or val_loss < min(self.val_losses):
>>>>>>> 12c0a976135ba658ad84c5b9fcf442122bab2980
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), join(model_path,
                                'best_', self.model_type,'_model.pt'))
            epochs.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, mean_epoch_loss))


if __name__ == '__main__':
    tr_df, val_df, test_df = read_data(tr_fp, val_fp, test_fp)
    sizes = [df.shape[0] for df in (tr_df, val_df, test_df)]
    full_df = pd.concat([tr_df, val_df, test_df])
    full_kg = torchkge.data_structures.KnowledgeGraph(full_df)
    tr_kg, val_kg, test_kg = full_kg.split_kg(sizes=sizes)
    te_mod = CustomTransModel(tr_kg, model_type = 'TransH')
    # he_mod = CustomBilinearModel(tr_kg, model_type = 'HolE')
    te_mod.set_sampler(samplerClass=BernoulliNegativeSampler, kg=tr_kg)
    te_mod.set_optimizer(optClass=Adam)
    te_mod.set_loss(lossClass=MarginLoss, margin=0.5)
    # Move everything to CUDA if available
    if cuda.is_available():
        print("Using cuda.")
        cuda.empty_cache()
        te_mod.model.cuda()
        te_mod.loss_fn.cuda()
    te_mod.train_model(2, val_kg)
