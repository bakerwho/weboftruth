import pandas as pd 
data_path = '100/svo_data_train_1000000.dat'
save_path = '100_old/svo_data_ts100_train_1000000.dat'
df = pd.read_csv(data_path, sep='\t')
df.columns = ['from', 'rel', 'to']
new_df = df[['from', 'to', 'rel']].copy()
new_df['true_positive'] = True
new_df.to_csv(save_path, index=False, sep='\t')

data_path = '100/svo_data_valid_50000.dat'
save_path = '100_old/svo_data_ts100_valid_50000.dat'
df = pd.read_csv(data_path, sep='\t')
df.columns = ['from', 'rel', 'to']
new_df = df[['from', 'to', 'rel']].copy()
new_df['true_positive'] = True
new_df.to_csv(save_path, index=False, sep='\t')

data_path = '100/svo_data_test_250000.dat'
save_path = '100_old/svo_data_ts100_test_250000.dat'
df = pd.read_csv(data_path, sep='\t')
df.columns = ['from', 'rel', 'to']
new_df = df[['from', 'to', 'rel']].copy()
new_df['true_positive'] = True
new_df.to_csv(save_path, index=False, sep='\t')
