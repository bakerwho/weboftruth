# weboftruth
Torch + bash scripts for training Knowledge Graph embeddings using Subject-Verb-Object triples

Two embedding spaces are created (one for Entities (Subjects/Objects) and one for Relationships/Verbs). Extensive use is made of the package `torch-kge`, built on PyTorch.

This was part of a coursework for CAPP 30255 at the University of Chicago by Aabir Abubaker Kar Adarsh Mathew

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


