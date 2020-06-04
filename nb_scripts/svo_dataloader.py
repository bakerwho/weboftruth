"""import pandas as pd
import pickle

import torch

from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
# from torchkge.utils.datasets import load_fb15k237
# from torchkge.utils.datasets import load_wikidatasets

from tqdm.autonotebook import tqdm

import torchkge
from torchkge.evaluation import LinkPredictionEvaluator


svo_data_path = './datasources/SVO-tensor-dataset/'
interim_dumps = './saved_obj/interim/'
saved_path = './saved_obj/'

# Loading Test, Valid, Train sets
df1 = pd.read_csv(svo_data_path + 'svo_data_train_1000000.dat',
                   sep='\t', header=None, names=['from', 'rel', 'to'])

df2 = pd.read_csv(svo_data_path + 'svo_data_valid_50000.dat',
                   sep='\t', header=None, names=['from', 'rel', 'to'])

df3 = pd.read_csv(svo_data_path + 'svo_data_test_250000.dat',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
"""

svo_kg = torchkge.data_structures.KnowledgeGraph(df = pd.concat([df1, df2, df3]))

with open(interim_dumps + 'svo_kg_full.pkl', 'wb') as f:
    pickle.dump(svo_kg, f)

svo_kg_train, svo_kg_valid, svo_kg_test = svo_kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

emb_dim = 250
lr = 0.0004
n_epochs = 10
b_size = 64
margin = 0.5

model = TransEModel(emb_dim, svo_kg_train.n_ent, svo_kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(margin)

if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(svo_kg_train)
dataloader = DataLoader(svo_kg_train, batch_size=b_size, use_cuda='all')

iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description(
        'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                              running_loss / len(dataloader)))

model.normalize_parameters()

"""
# Link prediction evaluation on validation set.
evaluator = LinkPredictionEvaluator(model, svo_kg_valid)
evaluator.evaluate(b_size=32, k_max=10)
evaluator.print_results()
"""

torch.save(model.state_dict(), saved_path+'transe_sdict.pt')
