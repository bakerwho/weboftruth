------------------------------------- 
-- SUBJECT-VERB-OBJECT TENSOR DATA --
-------------------------------------
A. Bordes -- License CC-BY-SA -- 2012
-------------------------------------

------------------
OUTLINE:
1. Introduction
2. Content
3. Data Format
4. Data Statistics
5. Data Collection
6. First Results
7. How to Cite
8. License
9. Contact
-------------------


1. INTRODUCTION:

The SUBJECT-VERB-OBJECT TENSOR DATA consists of a large collection of
triplet (subject, verb, direct object) extracted from Wikipedia, where
each member of the triplet is a single word belonging to the WordNet
lexicon (http://wordnet.princeton.edu):a noun for subject or direct
object and a verb for the last member. This data set can be seen as a
3-mode tensor depicting ternary relationships between nouns and
verbs. It has been first introduced in the paper "A Latent Factor
Model for Highly Multi-relational Data" published by R. Jenatton,
N. Le Roux, A. Bordes and G. Obozinski at the NIPS conference in 2012.


2. CONTENT:
The data archive contains 6 files:
  - README 1,6K
  - svo-nouns.lst 481K
  - svo-verbs.lst 56K
  - svo_data_test_250000.dat 3,8M
  - svo_data_train_1000000.dat 15M
  - svo_data_valid_50000.dat 779K

The 3 files svo_data_*.dat contain the triplets (training, validation
and test sets), while the two files svo-*.lst list the vocabulary.


3. DATA FORMAT

The two vocabulary files (svo-*.lst) contain one word per line: the
index of a noun is its line number if svo-nouns.lst, the index of a
verb if its line number in svo-verbs.lst. All indexes start at 1.

All svo_data_*.dat files contain one triplet per line, with 3 indexes
in a tab separated format. The first integer displays the index of the
subject in the nouns dictionary, the second the index of the verb in
the verbs dictionary and the last one the index of the object nouns
dictionary. Be careful that indexes of nouns (1st and 3rd columns) and
verbs (2nd columns) do not refer to the same dictionary.


4. DATA STATISTICS

The nouns dictionary size is 30,605 and the verbs one is 4,547. This
means that there are 4,547 possible relation types among nouns in this
data set.  The training set contains 1,000,000 triplets, the
validation set 50,000 and the test set 250,000.

All triplets are unique and we made sure that all words appearing in
the validation or test sets were occurring in the training set.

5. DATA COLLECTION:

Data was collected in two stages. First, the SENNA software (freely
available from http://ronan.collobert.com/senna/) was used to perform
part-of-speech tagging, chunking, lemmatization (lemmatization was
carried out using the NLTK toolkit, http://www.nltk.org) and semantic
role labeling on ~2,000,000 Wikipedia articles. Then, data was
filtered to only select sentences for which the semantic role labeling
structure was (subject, verb, direct object) with each term of the
triplet being a single word from the WordNet lexicon
(http://wordnet.princeton.edu). Subjects, verbs and direct objects are
all single terms from WordNet.

6. FIRST RESULTS

See the paper "A Latent Factor Model for Highly Multi-relational Data"
(available from the project page at http://goo.gl/TGYuh) for initial
results of link prediction with this data for 3 methods.

7. HOW TO CITE

When using this data, one should cite the original paper:
  @incollection{jenatton12,
    title = {A Latent Factor Model for Highly Multi-relational Data},
    author = {Rodolphe Jenatton and Nicolas {Le Roux} and Antoine Bordes and Guillaume Obozinski},
    booktitle = {Advances in Neural Information Processing Systems 25},
    editor = {P. Bartlett and F. Pereira and L. Bottou and {C.J.C} Burges and K. Weinberger},
    year = {2012}
  }

One should also point at the project page with either the long URL:
https://www.hds.utc.fr/everest/doku.php?id=en:lfmnips12 , or the short
one: http://goo.gl/TGYuh .

8. LICENSE:

The SUBJECT-VERB-OBJECT TENSOR DATA is made available under the
Creative Commons Attribution-ShareAlike 3.0 Unported License
(CC-BY-SA) whose full text can be found at
http://creativecommons.org/licenses/by-nc-sa/3.0/legalcode .

9. CONTACT

For all remarks or questions please contact Antoine Bordes: antoine
(dot) bordes (at) utc (dot) fr .



