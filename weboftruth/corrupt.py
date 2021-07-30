import pandas as pd
import torchkge
from os.path import join
from sklearn.model_selection import train_test_split

from weboftruth.utils import load_model

import sys

def corrupt_kg(input_kg, save_folder=None,
                sampler='BernoulliNegativeSampler',
                true_share=0.8, use_cuda=False, prefilename=''):
    """
    Input: KG structure
    Output:
        - KG object with true_share number of corrupted relations
        - Pandas DF to construct corrupted KG, with 'true_positive' flag indicating if triple is corrupted or not.
    """
    assert true_share>=0 and true_share<=1, 'Invalid true_share'
    if isinstance(sampler, str):
        sampler = getattr(torchkge.sampling, sampler)

    ent2ix, rel2ix = input_kg.ent2ix, input_kg.rel2ix
    ix2ent = {v:k for k, v, in ent2ix.items()}
    ix2rel = {v:k for k, v, in rel2ix.items()}

    def fact2txt(fact):
        s, o, v = fact
        try:
            s_, o_ = ix2ent[s], ix2ent[o]
            v_ = ix2rel[v]
            return s_, o_, v_
        except:
            #print(s, v, o)
            return None, None, None

    if true_share == 1:
        return input_kg
    else:
        kg_true, kg_to_corrupt = input_kg.split_kg(share = true_share)
        # setup true dataframe
        true_list = [fact2txt(kg_true[i]) for i in range(kg_true.n_facts)]
        true_df = pd.DataFrame(true_list, columns =['from', 'to', 'rel'])
        true_df['true_positive'] = True

        # setup corrupted dataframe
        kg_corrupted = sampler(kg_to_corrupt
                        ).corrupt_kg(batch_size=128, use_cuda=use_cuda,
                        which = 'main')
        corrupt_list = []
        for (s, o, v) in kg_corrupted:
            corrupt_list.append(fact2txt((s, o, v)))
        corrupt_df = pd.DataFrame(corrupt_list, columns =['from', 'to', 'rel'])
        corrupt_df['true_positive'] = False

        corrupt_kg_df = pd.concat([true_df, corrupt_df])
        corrupt_kg = torchkge.KnowledgeGraph(
                        df=corrupt_kg_df.drop(['true_positive'],
                        axis = 'columns'))

        if save_folder is not None:
            name = f'{prefilename}_ts={true_share}.dat'
            outfile = join(save_folder, name)
            corrupt_kg_df.to_csv(outfile, index=False, sep='\t')
            print(f'Writing ts={true_share} KnowledgeGraph to {outfile}')
        return corrupt_kg, corrupt_kg_df

if __name__=='__main__':
    pass
    # TODO: restructure this so that:
    # Inputs: filepath (CSV/DF), truth share, outfile name
    # OUTPUT: corrupted csv/df + write to disk
    """
    print(f"Datapath: {args.datapath}\nDataset: {args.dataset}\n")
    print(f"Sampler: {args.sampler}")
    print(f"Truth share: {args.ts}\nEmbedding dimension: {args.emb_dim}")

    sampler = eval(f"torchkge.sampling.{args.sampler}")

    dfs = wot.utils.get_simonepri_dataset_dfs(args.datapath, args.dataset)

    # optionally shuffle dataset
    if args.shuffle:
        tr_kg, val_kg, test_kg = wot.utils.reshuffle_trte_split(dfs)
    else:
        tr_kg, val_kg, test_kg = (wot.utils.df_to_kg(df) for df in dfs)

    for ts in [args.ts]:
        corrupt_kg(full_kg, save_folder=svo_paths[ts],
                    true_share=ts/100)
    """

"""
import weboftruth as wot
import torchkge
import pandas as pd
tr_fn, val_fn, test_fn = wot.utils.get_file_names(100)
tr_df, val_df, test_df = wot.utils.read_data(tr_fn, val_fn, test_fn,
                            wot.svo_paths[100])
full_df = pd.concat([tr_df, val_df, test_df])
full_kg = torchkge.data_structures.KnowledgeGraph(full_df)
for ts in [80, 50]:
    wot.corrupt.corrupt_kg(full_kg, save_folder=wot.svo_paths[ts],
                true_share=ts/100)
"""
