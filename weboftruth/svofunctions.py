import pandas as pd
import os
from os.path import join
import numpy as np

import torchkge
from torchkge import models

from datetime import datetime
from tabulate import tabulate

import sys
import argparse
parser = argparse.ArgumentParser()

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

def idx_dictionaries(path_to_entities = join(svo_data_path, 'svo-nouns.lst'),
                    path_to_relations = join(svo_data_path, 'svo-verbs.lst')):
    
    entity_dict = {}
    relation_dict = {}

    with open(path_to_entities, 'r') as f:
        for i, line in enumerate(f):
            elem_list = [elem.strip() for elem in line.split('__')]
            entity_name = elem_list[1][:(len(elem_list[1])-3)]
            entity_dict[i] = entity_name

    with open(path_to_relations, 'r') as f:
        for i, line in enumerate(f):
            elem_list = [elem.strip() for elem in line.split('__')]
            relation_name = elem_list[1][:(len(elem_list[1])-3)]
            relation_dict[i] = relation_name

    return entity_dict, relation_dict


