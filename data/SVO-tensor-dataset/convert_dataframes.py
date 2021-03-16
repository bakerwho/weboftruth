import pandas as pd 

#data_path = '80_old/svo_data_ts80_train_1000000.dat'
#save_path = '80/svo_data_ts80_train_1000000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

#data_path = '80_old/svo_data_ts80_valid_250000.dat'
#save_path = '80/svo_data_ts80_valid_250000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

#data_path = '80_old/svo_data_ts80_test_50000.dat'
#save_path = '80/svo_data_ts80_test_50000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

#data_path = '50_old/svo_data_ts50_train_1000000.dat'
#save_path = '50/svo_data_ts50_train_1000000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

#data_path = '50_old/svo_data_ts50_valid_250000.dat'
#save_path = '50/svo_data_ts50_valid_250000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

#data_path = '50_old/svo_data_ts50_test_50000.dat'
#save_path = '50/svo_data_ts50_test_50000.dat'
#df = pd.read_csv(data_path, sep='\t')
#new_df = df[['from', 'rel', 'to']].copy()
#new_df.to_csv(save_path, index=False, sep='\t', header=False)

data_path = '90_old/svo_data_ts90_train_1000000.dat'
save_path = '90/svo_data_ts90_train_1000000.dat'
df = pd.read_csv(data_path, sep='\t')
new_df = df[['from', 'rel', 'to']].copy()
new_df.to_csv(save_path, index=False, sep='\t', header=False)

data_path = '90_old/svo_data_ts90_valid_250000.dat'
save_path = '90/svo_data_ts90_valid_250000.dat'
df = pd.read_csv(data_path, sep='\t')
new_df = df[['from', 'rel', 'to']].copy()
new_df.to_csv(save_path, index=False, sep='\t', header=False)

data_path = '90_old/svo_data_ts90_test_50000.dat'
save_path = '90/svo_data_ts90_test_50000.dat'
df = pd.read_csv(data_path, sep='\t')
new_df = df[['from', 'rel', 'to']].copy()
new_df.to_csv(save_path, index=False, sep='\t', header=False)
