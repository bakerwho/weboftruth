import pandas as pd
import numpy as np 
from collections import Counter

path = '/home-nfs/tenzorok/weboftruth/data/SVO-tensor-dataset/100/svo_data_train_1000000.dat'

df = pd.read_csv(path, sep='\t')
df.columns = ['from', 'rel', 'to']
print('train')

#a = np.array(Counter(df['from']).values())
#print(np.sum(a))
print('unique FROM', len(set(df['from'])))
print('len',len(df), 'max val', np.max(df['from']))

print('unique TO', len(set(df['to'])))
print('len',len(df), 'max val', np.max(df['to']))

print('unique REL', len(set(df['rel'])))
print('len',len(df), 'max val', np.max(df['rel']))
