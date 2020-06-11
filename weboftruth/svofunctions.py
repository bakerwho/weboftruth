import pandas as pd
import os
from os.path import join
import numpy as np
import torch
import zipfile

import torchkge
from torchkge import models

from datetime import datetime
from tabulate import tabulate

import weboftruth as wot

## pathnames
# args.path = "~/weboftruth"
# args.path = "/project2/jevans/aabir/weboftruth/"
# args.path = "/Users/aabir/Documents/research/weboftruth"
################################################################
"""
#-p wot_path -e epochs -m model_type -small False
parser.add_argument("-p", "--path", dest="path",
                        default="/project2/jevans/aabir/weboftruth/",
                        help="path to weboftruth")
parser.add_argument("-ts", "--truthshare", dest="ts", default=100,
                        help="truth share of dataset", type=int)

args = parser.parse_args()

svo_data_path = join(args.path, 'data/SVO-tensor-dataset')
svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 80, 50]}

models_path = join(args.path, 'models')
"""
#get glove embeddings for svo

def idx_dictionaries(path_to_entities = join(wot.wotmodels.svo_data_path,
                    'svo-nouns.lst'),
                    path_to_relations = join(wot.wotmodels.svo_data_path,
                    'svo-verbs.lst')):
    entity_word_dict = {}
    relation_word_dict = {}
    with open(path_to_entities, 'r') as f:
        for i, line in enumerate(f):
            elem_list = [elem.strip() for elem in line.split('__')]
            entity_name = elem_list[1][:(len(elem_list[1])-3)]
            entity_word_dict[i] = entity_name
    with open(path_to_relations, 'r') as f:
        for i, line in enumerate(f):
            elem_list = [elem.strip() for elem in line.split('__')]
            relation_name = elem_list[1][:(len(elem_list[1])-3)]
            relation_word_dict[i] = relation_name
    return entity_word_dict, relation_word_dict


def get_glove_embeddings(path_to_glove, n_dim = 50):
    glove_dict = {}
    with zipfile.ZipFile(path_to_glove, 'r') as zip:
        files =  zip.namelist()
        for file in files:
            if str(n_dim) in file:
                with zip.open(file, 'r') as myfile:
                    for line in myfile:
                        embed_list = line.decode("utf-8").split()
                        glove_dict[str(embed_list[0])] = torch.FloatTensor(
                                        [float(i) for i in embed_list[1:]])
    return glove_dict

def svo_glove(svo_entity_dict, svo_rel_dict, glove_dict, n_dim):
    svo_entity_glovedict = {}
    svo_relation_glovedict = {}
    print(len(glove_dict.keys()))
    i, j = 0, 0
    for (k, val) in svo_entity_dict.items():
        val_list = val.split('_')
        val_embed = torch.zeros(n_dim, dtype=torch.float64)
        for unigram in val_list:
            try:
                val_embed = val_embed.add(glove_dict[unigram])
                i+=1
            except KeyError:
                val_embed = val_embed.add(torch.zeros(n_dim, dtype=torch.float64))
                j+=1
        val_embed = val_embed.div(len(val_list))
        svo_entity_glovedict[k] = val_embed
    for (k, val) in svo_rel_dict.items():
        val_list = val.split('_')
        val_embed = torch.zeros(n_dim, dtype=torch.float64)
        for unigram in val_list:
            try:
                val_embed = val_embed.add(glove_dict[unigram])
            except KeyError:
                val_embed = val_embed.add(torch.zeros(n_dim, dtype=torch.float64))
        val_embed = val_embed.div(len(val_list))
        svo_relation_glovedict[k] = val_embed
    return svo_entity_glovedict, svo_relation_glovedict
