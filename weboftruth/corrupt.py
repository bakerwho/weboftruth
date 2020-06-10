import pandas as pd
import torchkge
from os.path import join
from sklearn.model_selection import train_test_split

from weboftruth.wotmodels import *
from weboftruth.utils import load_model

def corrupt_kg(input_kg, save_folder=None,
                sampler=torchkge.sampling.BernoulliNegativeSampler,
                true_share=0.8, use_cuda=False):
    """
    Input: KG structure
    Output:
        - KG object with true_share number of corrupted relations
        - Pandas DF to construct corrupted KG, with 'true_positive' flag indicating if triple is corrupted or not.
    """
    assert true_share>=0 and true_share<=1, 'Invalid true_share'
    if true_share == 1:
        return input_kg
    else:
        kg_true, kg_to_corrupt = input_kg.split_kg(share = true_share)
        true_list = [kg_true[i] for i in range(kg_true.n_facts)]
        kg_corrupted = sampler(kg_to_corrupt
                        ).corrupt_kg(batch_size = 128, use_cuda = use_cuda,
                        which = 'main')
        corrupt_list = []
        for i in range(kg_to_corrupt.n_facts):
            corrupt_list.append((kg_corrupted[0][i].item(),
                                    kg_corrupted[1][i].item(),
                                    kg_to_corrupt[i][2]))
        corrupt_df = pd.DataFrame(corrupt_list, columns =['from', 'to', 'rel'])
        corrupt_df['true_positive'] = False
        true_df = pd.DataFrame(true_list, columns =['from', 'to', 'rel'])
        true_df['true_positive'] = True
        corrupt_kg_df = pd.concat([true_df, corrupt_df])
        #corrupt_kg = torchkge.data_structures.KnowledgeGraph(
        #                df=corrupt_kg_df.drop(['true_positive'],
        #                axis = 'columns'))
        out_dfs = {}
        out_dfs['train'], df2 = train_test_split(corrupt_kg_df,
                                                    train_size=int(1e6))
        out_dfs['valid'], out_dfs['test'] = train_test_split(df2, train_size=250000)
        config = int(true_share*100)
        sizedict = dict(zip(['train', 'valid', 'test'],
                [1000000, 250000, 50000])).items()
        out_kgs = []
        for setname, df in out_dfs.items():
            if savepath is not None:
                name = f'svo_data_ts{config}_{setname}_{sizedict[setname]}.dat'
                outfile = join(save_folder, name)
                df.to_csv(outfile, index=False)
                print(f'Writing {setname} KnowledgeGraph to {outfile}')
            out_kgs.append(torchkge.data_structures.KnowledgeGraph(
                            df=df.drop(['true_positive'],
                            axis = 'columns')))
        # tr_kg, val_kg, test_kg = out_kgs
        return out_kgs

if __name__=='__main__':
    for ts in [80, 50]:
        tr_fn, val_fn, test_fn = wot.utils.get_file_names(ts)
        tr_df, val_df, test_df = read_data(tr_fn, val_fn, test_fn,
                                    svo_paths[100])
        full_df = pd.concat([tr_df, val_df, test_df])
        full_kg = torchkge.data_structures.KnowledgeGraph(full_df)
        corrupt_kg(full_kg, save_folder=svo_paths[ts],
                    true_share=ts)
