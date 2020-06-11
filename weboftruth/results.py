import weboftruth as wot

from os.path import join

from sklearn.linear_model import LogisticRegression

import time

svofolder = '/Users/aabir/Documents/research/weboftruth/data/SVO-tensor-dataset'
ent_word_dict, rel_word_dict = wot.idx_dictionaries(
                            path_to_entities=join(svofolder, 'svo-nouns.lst'),
                            path_to_relations=join(svofolder, 'svo-verbs.lst'))
glovefolder = '/Users/aabir/Documents/research/weboftruth/data/glove.6B.zip'

for n_dim in [50, 100, 200, 300]:
    glove_dict = wot.get_glove_embeddings(glovefolder, n_dim=n_dim)
    ent_glove, rel_glove = wot.svo_glove(ent_word_dict, rel_word_dict, glove_dict, n_dim)

    print(f"\n\nloaded glove for dimension {n_dim}")

    tr_fn, val_fn, test_fn = wot.utils.get_file_names(50)
    tr_fp = join(svofolder, '50', tr_fn)
    x_tr, y_tr = wot.get_svo_glove_embeddings(tr_fp, ent_glove, rel_glove)

    test_fp = join(svofolder, '50', test_fn)
    x_te, y_te = wot.get_svo_glove_embeddings(test_fp, ent_glove, rel_glove)

    print("loaded train and test data")

    t1 = time.time()

    model = wot.train_sklearnmodel(LogisticRegression, x_tr, y_tr, solver='sag', max_iter=400)
    wot.evaluate_model(model, x_te, y_te)

    t2 = time.time()
    print(f"\ttime taken: {t2-t1} s")
