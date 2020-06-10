from setuptools import setup

setup(
   name='weboftruth',
   version='0.1',
   description='context-aware truth detection using knowledge graph embeddings',
   author='AAK/AM',
   author_email='aabir@uchicago.edu',
   packages=['weboftruth'],  #same as name
   install_requires=['torchkge', 'tabulate', 'pandas', 'numpy'], #external packages as dependencies
)
