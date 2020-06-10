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
