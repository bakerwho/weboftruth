import pandas as pd
import torchkge
from os.path import join
from sklearn.model_selection import train_test_split

from weboftruth.utils import load_model, kg_to_df

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
    if true_share == 1:
        return input_kg

    if isinstance(sampler, str):
        sampler = getattr(torchkge.sampling, sampler)

    ent2ix, rel2ix = input_kg.ent2ix, input_kg.rel2ix
    ix2ent = {v:k for k, v, in ent2ix.items()}
    ix2rel = {v:k for k, v, in rel2ix.items()}

    def fact2txt(s, o, v):
        try:
            s_, o_ = ix2ent[s], ix2ent[o]
            v_ = ix2rel[v]
            return s_, o_, v_
        except:
            #print(s, v, o)
            print(f'Potential error: ix2ent or ix2rel failed for {(s, o, v)}')
            return s, o, v

    kg_true1, kg_tofalse1 = input_kg.split_kg(0.5)
    tot_ct, t_ct, f_ct = input_kg.n_facts, kg_true1.n_facts, kg_tofalse1.n_facts

    diff = abs(t_ct-f_ct)//2

    if t_ct > f_ct:
        #print(f"swap {diff} facts from true to false")
        kg_true, kg_tofalse2 = kg_true1.split_kg(sizes=[t_ct-diff, diff])
        df_tofalse1, df_tofalse2= kg_to_df(kg_tofalse1), kg_to_df(kg_tofalse2)
        df_tofalse = pd.concat([df_tofalse1, df_tofalse2])
        df_true = kg_to_df(kg_true)
    else:
        #print(f"swap {diff} facts from false to true")
        kg_tofalse, kg_true2 = kg_tofalse1.split_kg(sizes=[f_ct-diff, diff])
        df_true1, df_true2 = (wot.utils.kg_to_df(x) for x in (kg_true1, kg_true2))
        df_true = pd.concat([df_true1, df_true2])
        df_tofalse = kg_to_df(kg_tofalse)

    kg_true, kg_tofalse = [torchkge.KnowledgeGraph(df=x)
                                for x in [df_true, df_tofalse]]
    df_true['true_positive'] = True
    df_tofalse['true_positive'] = False

    diff2 = df_true.shape[0] - df_tofalse.shape[0]

    assert abs(diff2)<2, f'Mismatch {diff2} between true and false facts'

    # setup corrupted dataframe
    neg_heads, neg_tails = sampler(kg_tofalse
                    ).corrupt_kg(batch_size=128, use_cuda=use_cuda,
                    which = 'main')
    corrupt_list = []
    relations = kg_tofalse.relations.tolist()
    for i, (nh, nt) in enumerate(zip(neg_heads.tolist(),
                                     neg_tails.tolist())):
        corrupt_list.append(fact2txt((nh, nt, relations[i])))
    df_false = pd.DataFrame(corrupt_list, columns =['from', 'to', 'rel'])
    df_false['true_positive'] = False

    corrupted_df = pd.concat([df_true, df_false])
    corrupted_kg = torchkge.KnowledgeGraph(
                    df=corrupted_df.drop(['true_positive'],
                    axis='columns'))

    if save_folder is not None:
        name = f'{prefilename}_ts={true_share}.dat'
        outfile = join(save_folder, name)
        corrupted_df.to_csv(outfile, index=False, sep='\t')
        print(f'Writing ts={true_share} KG (neg sampler: {sampler}) to {outfile}')
    return corrupted_kg, corrupted_df

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
