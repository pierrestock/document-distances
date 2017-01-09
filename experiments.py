import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from time import time
from collections import Counter
from compute_distance import distance
from compute_cost_matrix import cost_matrix

# path to data
data_path = "/home/ubuntu/pcs/data/"

# load train/test data
docs_train = []
category_train = []
with open(data_path + "reuters_train.txt", encoding='utf-8') as f:
    for new in f:
        category, left = new.split("\t")
        text = left.split(" reuter")[0]
        docs_train = docs_train + [text]
        category_train = category_train + [category]

docs_test = []
category_test = []
with open(data_path + "reuters_test.txt", encoding='utf-8') as f:
    for new in f:
        category, left = new.split("\t")
        text = left.split(" reuter")[0]
        docs_test = docs_test + [text]
        category_test = category_test + [category]

# conversion into np arrays, needed later
docs_train = np.asarray(docs_train[:1000])
category_train = np.asarray(category_train[:1000])
docs_test = np.asarray(docs_test[:200])
category_test = np.asarray(category_test[:200])

# helper function
def compute_knn_pred(pred, k):
    counter = Counter(pred[:k])
    return max(counter, key=counter.get)

#--------------- influence of lambda  ---------------#
lambda_list = np.logspace(-1,2,10)
k_list = [1, 3, 5, 10, 20]
niter = 100
test_size = len(docs_test)
err = np.zeros([len(lambda_list), len(k_list)])

for i in range(len(lambda_list)):
    tm = time()
    lambd = lambda_list[i]
    print("Setting lambda to %.2f" %lambd)

    # loop in the test set
    for j in range(test_size):
        doc_to_test = docs_test[j]
        target = category_test[j]
        D = distance([doc_to_test], docs_train, lambd, niter, data_path)
        idx = np.argsort(D)
        pred = category_train[idx]
        print(D[:3])
        err[i, :] += [(compute_knn_pred(pred, k_curr) != target) for k_curr in k_list]

    # aggregate results
    err[i, :] = err[i, :] / test_size
    print("Elapsed = %.2f" %(time() - tm))

pkl.dump(err, open(data_path + "err_lambda.p", 'wb'))

 #--------------- influence of the embedding norm  ---------------#
size = [1000]
niter = 100
p_list = np.logspace(0, 1, 10)
k_list = [1, 3, 5, 10, 20]
test_size = len(docs_test)
err = np.zeros([len(p_list), len(k_list)])

# compute cost matrixes
C_names, key_names = cost_matrix(size, p_list, data_path)
#C_names = ["C_most_common_1000_" + str(p) + ".p" for p in p_list]
#key_names = ["keys_most_common_1000_"  + str(p) + ".p" for p in p_list]

for i in range(len(p_list)):
    tm = time()
    p = p_list[i]
    print("Setting p to %.2f" %p)

    # loop in the test set
    for j in range(test_size):
        doc_to_test = docs_test[j]
        target = category_test[j]
        D = distance([doc_to_test], docs_train, 100, niter, data_path, C_name = C_names[i], keys_name = key_names[i])
        idx = np.argsort(D)
        pred = category_train[idx]
        print(D[:3])
        err[i, :] += [(compute_knn_pred(pred, k_curr) != target) for k_curr in k_list]

    # aggregate results
    err[i, :] = err[i, :] / test_size
    print("Elapsed = %.2f" %(time() - tm))

pkl.dump(err, open(data_path + "err_p.p", 'wb'))


# --------------- infuence of the size of the dictionary ---------------#
# influence of the embedding norm
size_list = [1000, 2000, 3000, 5000, 1000]
niter = 100
p = [2]
k_list = [1, 3, 5, 10, 20]
test_size = len(docs_test)
err = np.zeros([len(size_list), len(k_list)])

# compute cost matrixes
C_names, key_names = cost_matrix(size_list, p, data_path)
#C_names = ["C_most_common_1000_" + str(p) + ".p" for p in p_list]
#key_names = ["keys_most_common_1000_"  + str(p) + ".p" for p in p_list]

for i in range(len(size_list)):
    tm = time()
    size = size_list[i]
    print("Setting size to %d" %size)

    # loop in the test set
    for j in range(test_size):
        doc_to_test = docs_test[j]
        target = category_test[j]
        D = distance([doc_to_test], docs_train, 100, niter, data_path, C_name = C_names[i], keys_name = key_names[i])
        idx = np.argsort(D)
        pred = category_train[idx]
        print(D[:3])
        err[i, :] += [(compute_knn_pred(pred, k_curr) != target) for k_curr in k_list]

    # aggregate results
    err[i, :] = err[i, :] / test_size
    print("Elapsed = %.2f" %(time() - tm))

pkl.dump(err, open(data_path + "err_size.p", 'wb'))
