# How to compute document similarities ? 

This repository is a Tensorflow implementation of a project consisting in computing document similarities. The full pdf report (__report__ folder) details is here to (try to) answer any question you may have. 

1. [Setup](#setup)

2. [Playing around](#playing-around)

3. [Going further](#going-further)

 
## Setup 

If you wish to use this code, you will have to install the following package:
```
pip install gensim 
```
The pretrained word2vec model is available [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (1.5 GB !)

You can take a look at the iPython Notebook __toy_examples.ipynb__ that contains a very brief description of the algorithm, some fun properties of the word2vec metric and studies the influence of some parameters on a toy example. The toy data files are already in the __data__ folder.  

## Playing around

The algorithm is implemented in Python 3.5 using Tensorflow 0.11 in the file __compute_distance.py__. If you wish to run the __experiments.py__ file containing all the experiments made during this project, you will have to either 

* Use the precomputed cost matrix __C_most_common_1000_2.p__ and the associated keys __keys_most_common_1000_2.p__

## Going further

If you are interested in this topic, you can read the full pdf report (__report__ folder) that details some theoretical aspects, the methodology and the experimental results on large datasets. A bibliography is included if you want to go deeper on the theoretical side. 
