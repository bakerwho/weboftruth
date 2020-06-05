import pandas as pd
import pickle
import os
from os.path import join

import torchkge
import torch
from torch import cuda
from torch.optim import Adam

from torchkge.models import Model, TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader

from tqdm.autonotebook import tqdm

# fix this
#wot_path = "~/weboftruth"
wot_path = "/Users/aabir/Documents/research/weboftruth"
svo_data_path = join(wot_path, 'SVO-tensor-dataset')
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

class CustomModel(TransEModel):
    def __init__(self, kg, **kwargs):
        self.kg = kg
        self.emb_dim = kwargs.pop('emb_dim', 250)
        self.lr = kwargs.pop('lr', 0.0004)
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.b_size = kwargs.pop('b_size', 64)
        self.diss_type = kwargs.pop('diss_type', 'L2')
        super(CustomModel, self).__init__(self.emb_dim, kg.n_ent, kg.n_rel,
                            dissimilarity_type=self.diss_type)
        if cuda.is_available():
            cuda.empty_cache()
            self.cuda()
        try:
            self.dataloader = DataLoader(kg, batch_size=self.b_size, use_cuda='all')
        except AssertionError:
            self.dataloader = DataLoader(kg, batch_size=self.b_size)

    def set_optimizer(self, optClass=Adam, **kwargs):
        self.optimizer = optClass(self.parameters(), lr=self.lr,
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
            pos, neg = self.forward(h, t, n_h, n_t, r)
            loss = self.loss_fn(pos, neg)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.normalize_parameters()
        return running_loss/i

    def train_model(self, n_epochs):
        self.normalize_parameters()
        epochs = tqdm(range(n_epochs), unit='epoch')
        for epoch in epochs:
            mean_epoch_loss = self.one_epoch()
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, mean_epoch_loss))


if __name__ == '__main__':
    tr_df, val_df, test_df = read_data(tr_fp, val_fp, test_fp)
    sizes = [df.shape[0] for df in (tr_df, val_df, test_df)]
    full_df = pd.concat([tr_df, val_df, test_df])
    full_kg = torchkge.data_structures.KnowledgeGraph(full_df)
    tr_kg, val_kg, test_kg = full_kg.split_kg(
                                            sizes=sizes)
    te_mod = CustomModel(full_kg)
    te_mod.set_sampler(samplerClass=BernoulliNegativeSampler, kg=tr_kg)
    te_mod.set_optimizer(optClass=Adam)
    te_mod.set_loss(lossClass=MarginLoss, margin=0.5)
    te_mod.train_model(100)
