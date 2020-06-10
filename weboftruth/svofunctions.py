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


def get_glove_embeddings(path_to_glove, svo_entity_dict, svo_rel_dict, n_dim=50):

    glove_dict = {}
    svo_entity_glovedict = {}
    svo_relation_glovedict = {}

    with zipfile.ZipFile(path_to_glove, 'r') as zip:

        files =  zip.namelist()

        for file in files:
            if str(n_dim) in files:
                glove_file = file

        with zip.open(glove_file, 'r') as myfile:
            for line in myfile:
                embed_list = line.split()
                glove_dict[embed_list[0]] = torch.Tensor([float(i) for i in embed_list[1:]], dtype = torch.float64)

    for k, val in svo_entity_dict.values():
        val_list = val.split('_')
        val_embed = torch.zeros(n_dim, dtype=torch.float64)
        for unigram in val_list:
            try:
                val_embed.add(glove_dict[unigram])
            except KeyError:
                val_embed.add(glove_dict['unk'])
        val_embed = val_embed.div(len(val_list))
        svo_entity_glovedict[k] = val_embed

    for k, val in svo_rel_dict.values():
        val_list = val.split('_')
        val_embed = torch.zeros(n_dim, dtype=torch.float64)
        for unigram in val_list:
            try:
                val_embed.add(glove_dict[unigram])
            except KeyError:
                val_embed.add(glove_dict['unk'])
        val_embed = val_embed.div(len(val_list))
        svo_relation_glovedict[k] = val_embed

    return svo_entity_glovedict, svo_relation_glovedict


"""print(myfile.read())
# printing all the contents of the zip file
zip.printdir()

# extracting all the files
print('Extracting all the files now...')
zip.extractall()
print('Done!') """
