# weboftruth

**weboftruth** is a project to use deep representation learning to ascertain the credibility of a statement given past context. It is a proof-of-concept for one approach to algorithmic fake news detection.

This repo contains Python and bash scripts for training Knowledge Graph embeddings using Subject-Verb-Object triples.

Two embedding spaces are created (one for Entities (Subjects/Objects) and one for Relationships/Verbs). Extensive use is made of the package `torchkge` that implements KGE algorithms like TransE. It is built on PyTorch.

## Results:
- We trained Entity and Relationship embeddings for a cleaned dataset of SVO triples constructed from Wikipedia sentences.
- We trained three configurations - embeddings of dimension 50, 100 and 200.
- A binary classifier (logistic regression) on concatenated SVO embeddings recorded 79% accuracy at telling **true** triples apart from **false** (negatively sampled) ones

This was part of a coursework for CAPP 30255 at the University of Chicago by Aabir Abubaker Kar Adarsh Mathew

## Filestructure:

- data

  contains data sources used, specifically the SVO dataset of subject-verb-object triples from Wikipedia, and our constructed 'partially true' datasets
- logs

  execution logs for RCC and AWS runs
- notebooks

  Jupyter notebooks used for early prototyping
- shscripts

  shell scripts used for RCC
- weboftruth

  Python code organized into a package. Borrows heavily from torch and torchkge
  - utils.py: miscellaneous useful functions for loading models, data etc.
  - corrupt.py: code for generating the partially true datasets and writing to disk
  - svofunctions.py: tools to translate triples to indices and vice versa
  - wotmodels.py: tools to build and train models from torchkge
  - evaluator.py: functions to evaluate the performance of embeddings as features in truth prediction


