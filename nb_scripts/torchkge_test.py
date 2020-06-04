#!/usr/bin/env python
# coding: utf-8

# In[27]:


from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_fb15k237
from torchkge.utils.datasets import load_wikidatasets

from tqdm.autonotebook import tqdm


# In[40]:


# Load dataset
kg_train, _, _ = load_fb15k237()
# kg_train_hum, _, _  = load_wikidatasets('humans', limit_ = 20, data_home = 'C:/Users/Mathew/torchkge_data')


# In[17]:


# Define some hyper-parameters for training
emb_dim = 250
lr = 0.0004
n_epochs = 1000
b_size = 32768
margin = 0.5


# In[18]:


# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(margin)


# In[43]:


# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()


# In[20]:


# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

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

