import numpy as np
import pandas as pd
import time
import pprint

import os
from os.path import join

import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags

nltk.download('punkt')

import spacy
from spacy import displacy
from collections import Counter
try:
    nlp = spacy.load('en_core_web_sm')
except:
    pass

#ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

now_folder = '/project2/jevans/aabir/NOWwhat/'
d_folder = '/project2/jevans/aabir/NOWwhat/xdata'
in_folder = join(d_folder, 'in_data')
us_folder = join(d_folder, 'us_data')

corpus_folder = join(d_folder, 'corpus')
result_folder = join(now_folder, 'resultdata')

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def nltk_ner(ex):
    sent = preprocess(ex)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    pprint(iob_tagged)
    ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
    print(ne_tree)
    return cs, iob_tagged, ne_tree


def spacy_ner(ex, output=False, cond=None):
    """Calls Spacy's en_core_web_sm NER model on text input
        inputs: ex (str), text input
                output (bool), whether to print NER output
        outputs: doc, the object returned by Spacy
    """
    parsed = nlp(ex)
    if cond is None:
        cond = lambda x : True
    if output and cond(parsed):
        pprint([(X.text, X.label_) for X in parsed.ents])
        pprint([(X, X.ent_iob_, X.ent_type_) for X in parsed])
    return parsed

def ner_on_file(filepath, outfilepath):
    data = {'ent_list':[], 'rel_text':[]}
    with open(filepath, 'r') as infile:
        for doc in infile:
            for line in sent_tokenize(doc):
                parsed = spacy_ner(line, output=False)
                if len(parsed.ents) == 0:
                    continue
                ents = ';'.join([X.text for X in parsed.ents])
                rel = ' '.join([X.text for X in parsed if X.text not in ents])
                data['ent_list'].append(ents)
                data['rel_text'].append(rel)
    df = pd.DataFrame(data)
    df.to_csv(outfilepath, index=False)
    print(f'month {filepath} NER complete; stored in {outfilepath}')

def apply_ner(in_folder, out_folder, cond=None):
    files = sorted([i for i in os.listdir(in_folder
                                ) if os.path.isfile(join(in_folder, i))])
    if cond is None:
        cond = lambda x : True
    for file in files:
        if not cond(file):
            continue
        month = file.split('.')[-2][-8:-3]
        fn = file.split('.')[0]
        ner_on_file(join(in_folder, file), join(out_folder, fn+'.csv'))

out_folder = join(d_folder, 'ner_csv')
os.makedirs(out_folder, exist_ok=True)
apply_ner(us_folder, out_folder, cond=lambda x: '16-11' in x)


# t = spacy_ner("Narendra Modi on Friday moved the Ministry of External Affairs to criticize China on the Hong Kong Crisis", output=True)
