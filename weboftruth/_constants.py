from os.path import join

import weboftruth

svo_data_path = join('.', 'data', 'SVO-tensor-dataset')
svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 80, 50]}
models_path = join('.', 'models')

def reset_paths(path):
    weboftruth.svo_data_path = join(path, 'data', 'SVO-tensor-dataset')
    weboftruth.svo_paths = {k:join(svo_data_path, str(k)) for k in [100, 80, 50]}
    weboftruth.models_path = join(path, 'models')
