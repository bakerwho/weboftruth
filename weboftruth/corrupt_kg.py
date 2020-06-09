import pandas as pd
import torchkge
from os.path import join

def kg_corrupt(input_kg, save_path, sampler=torchkge.sampling.BernoulliNegativeSampler, true_share = 0.8, use_cuda = True):
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
        cp_corrupt_df = pd.DataFrame(corrupt_list, columns =['from', 'to', 'rel'])
        cp_corrupt_df['true_positive'] = False
        true_df = pd.DataFrame(true_list, columns =['from', 'to', 'rel'])
        true_df['true_positive'] = True
        corrupt_kg_df = pd.concat([true_df, cp_corrupt_df])
        corrupt_kg = torchkge.data_structures.KnowledgeGraph(df = corrupt_kg_df.drop(['true_positive'], axis = 'columns'))
        corrupt_kg_df.to_csv(join(save_path,
                                f'corrupt_kg_df_{int(true_share*100)}.dat'),
                                index = False)
        return corrupt_kg
