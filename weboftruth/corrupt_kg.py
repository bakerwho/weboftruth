import pandas as pd 
import torchkge 

def kg_corrupt(input_kg, sampler=torchkge.sampling.BernoulliNegativeSampler, true_share = 0.8, use_cuda = True):
    """
    Input: KG structure
    Output: 
        - KG object with true_share number of corrupted relations
        - Pandas DF to construct corrupted KG, with 'true_positive' flag indicating if triple is corrupted or not.
    """
    if true_share == 1:
        return input_kg
    else:
        input_kg_tp, input_kg_cp = input_kg.split_kg(share = true_share)
        tp_list = [input_kg_tp[i] for i in range(input_kg_tp.n_facts)]
        tp_df = pd.DataFrame(tp_list, columns =['from', 'to', 'rel'])
        tp_df['true_positive'] = True
        input_kg_cp_corrupt = sampler(input_kg_cp).corrupt_kg(batch_size = 128, use_cuda = use_cuda, which = 'main')
        cp_list = []
        for i in range(input_kg_cp.n_facts):
            cp_list.append((input_kg_cp_corrupt[0][i].item(), input_kg_cp_corrupt[1][i].item(), input_kg_cp[i][2]))
        cp_corrupt_df = pd.DataFrame(cp_list, columns =['from', 'to', 'rel'])
        cp_corrupt_df['true_positive'] = False
        corrupt_kg_df = pd.concat([tp_df, cp_corrupt_df])
        corrupt_kg = torchkge.data_structures.KnowledgeGraph(df = corrupt_kg_df.drop(['true_positive']))
        ########### FIX THIS! ################
        corrupt_kg_df.to_csv("xxxxx.dat", index = False)
        ########### FIX THIS! ################
        return corrupt_kg
