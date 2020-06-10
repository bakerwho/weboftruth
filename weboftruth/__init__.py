import weboftruth.load_model
from weboftruth.load_model import *

import weboftruth.train_save_model
from weboftruth.train_save_model import *

import weboftruth.corrupt
from weboftruth.corrupt import *

import weboftruth.svofunctions
from weboftruth.svofunctions import *

import weboftruth.evaluator
from weboftruth.evaluator import *

import argparse
parser = argparse.ArgumentParser()

import os

cwd = os.getcwd()

svo_data_path = join(cwd, 'data', 'SVO-tensor-dataset')
svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 80, 50]}

models_path = join(cwd, 'models')
