# How to compute document similarities ? 

This repo is a GPU-ready Tensorflow implementation of a project consisting in computing document similarities. The full pdf report (__report__ folder) summarizes the project much more extensively if you have any questions.

1. [Setup](#setup)

2. [Playing around](#playing-around)

3. [Going further](#going-further)

 
## Setup 

If you wish to use this code, you will have to install the following package:
```
pip install gensim 
```
The pretrained word2vec model is available [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (1.5 GB !). You will need it for the toy examples and for computing some customized cost matrixes. 

You can take a look at the iPython Notebook __toy_examples.ipynb__ that contains a very brief description of the algorithm, some fun properties of the word2vec metric and studies the influence of some parameters on a toy example. The toy data files are already in the __data__ folder.  

## Playing around

The algorithm is implemented in Python 3.5 using Tensorflow 0.11 in the file __compute_distance.py__. If you wish to run the __experiments.py__ file containing all the experiments made during this project, you will have to either 

* Use the precomputed cost matrix __C_most_common_1000_2__ and the associated keys __keys_most_common_1000_2__
* Compute it yourself by chosing the number of words to include and the order of the norm in the embedding space by using the cost_matrix function defined in __compute_cost_matrix.py__ 

The experiments may take time to run (especially on a CPU), I ran them on a NVIDIA K80 using AWS.

## Going further

If you are interested in this topic, you can read the full pdf report (__report__ folder) that details some theoretical aspects, the methodology and the experimental results on large datasets. A bibliography is included if you want to dig deeper on the theoretical side. 
