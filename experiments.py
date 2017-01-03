import pickle as pkl
from distance_gpu import distance_gpu

# path to data
data_path = "/home/ubuntu/pcs/data/"

# load train/test data
reuters_train = []
category_train = []
with open(data_path + "reuters_train.txt", encoding='utf-8') as f:
    for new in f:
        category, left = new.split("\t")
        text = left.split(" reuter")[0]
        reuters_train = reuters_train + [text]
        category_train = category_train + [category]

reuters_test = []
category_test = []
with open(data_path + "reuters_test.txt", encoding='utf-8') as f:
    for new in f:
        category, left = new.split("\t")
        text = left.split(" reuter")[0]
        reuters_test = reuters_test + [text]
        category_test = category_test + [category]

# test 1 (must be OK))
D = distance_gpu(reuters_test[:1],reuters_train[:1], 1, 100, data_path, verbose = 1)

# test 2 (should be OK)
D = distance_gpu(reuters_test[:1],reuters_train[:1000], 1, 100, data_path, verbose = 1)

# test 3 (memory issues?)
D = distance_gpu(reuters_test[:1],reuters_train, 1, 100, data_path, verbose = 1)

# save results
pkl.dump(D, open(data_path + "D_matrix.p", 'wb'))
