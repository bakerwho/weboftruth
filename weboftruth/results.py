import weboftruth as wot

from os.path import join

from sklearn.linear_model import LogisticRegression

n_dim = 300
folder = join(wot.svo_data_path, 'glove.6B')
ent_word_dict, rel_word_dict = wot.idx_dictionaries()
glove_dict = wot.get_glove_embeddings(folder, n_dim=n_dim)
ent_glove, rel_glove = wot.svo_glove(ent_word_dict, rel_word_dict, glove_dict, n_dim)

tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
tr_fp = join(wot.svo_paths[50], tr_fn)
x_tr, y_tr = get_svo_glove_embeddings(tr_fp, ent_glove, rel_glove)

test_fp = join(paths[50], test_fn)
x_te, y_te = get_svo_glove_embeddings(test_fp, ent_glove, rel_glove)

model = wot.train_sklearnmodel(LogisticRegression, x_tr, y_tr)
wot.evaluate_model(model, x_te, y_te)
