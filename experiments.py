import numpy as np
import pickle as pkl
from time import time

from collections import Counter
from compute_distance import distance

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

# conversion stuff, needed later
docs_train = np.asarray(docs_train[:1000])
category_train = np.asarray(category_train[:1000])
docs_test = np.asarray(docs_test[:200])
category_test = np.asarray(category_test[:200])

tm = time()
D = distance(docs_test[:1], docs_train, 10, 100, data_path, verbose = 1)
print("1 ---", time() - tm)
tm = time()
D = distance(docs_test[:10], docs_train, 10, 100, data_path, verbose = 1)
print("10 ---", time() - tm)
tm = time()
D = distance(docs_test[:50], docs_train, 10, 100, data_path, verbose = 1)
print("50 ---", time() - tm)
tm = time()
D = distance(docs_test[:100], docs_train, 10, 100, data_path, verbose = 1)
print("100 ---", time() - tm)
tm = time()
D = distance(docs_test, docs_train, 10, 100, data_path, verbose = 1)
print("200 ---", time() - tm)
#
# # helper function
# def compute_knn_pred(pred, k):
#     counter = Counter(pred[:k])
#     return max(counter, key=counter.get)
#
# # influence of lambda
# lambda_list = np.logspace(-1,2,10)
# niter = 100
# results = []
#
# for i in range(len(lambda_list)):
#     err = 0
#     tm = time()
#     lambd = lambda_list[i]
#     print("Setting lambda to %.2f" %lambd)
#
#     # loop in the test set
#     for j in range(len(docs_test)):
#         doc_to_test = docs_test[j]
#         target = category_test[j]
#         D = distance([doc_to_test], docs_train, lambd, niter, data_path)
#         idx = np.argsort(D)
#         pred = category_train[idx]
#         err += (compute_knn_pred(pred, 1) == target)
#
#     # aggregate results
#     results = results + [err/len(docs_test)]
#     print("Elapsed = %.2f, error = %.2f" %(time() - tm, err))
#
# print(results)
# plt.figure(figsize = (8,6))
# plt.plot(results)
# plt.xticks(np.arange(10), np.around(lambda_list, 2))
# plt.xlabel("$\lambda$")
# plt.ylabel("1-nn Classification error")
# #plt.ylim([0.34, 0.45])
# plt.savefig(data_path + "lambda.png")
# plt.show()

# influence of the embedding norm
# size = 5000
# p_list = np.logspace(-1, 2, 10)
#
# # compute cost matrixes
# C_names, key_names = cost_matrix(size, p_list, data_path)
#
# for i in range(len(p_list)):
#     err = 0
#     lambd = lambda_list[i]
#     print("Setting lambda to %.2f" %lambd)
#
#     # loop in the test set
#     for j in range(len(docs_test)):
#         doc_to_test = docs_test[j]
#         target = category_test[j]
#         D = distance([doc_to_test], docs_train, lambd, niter, data_path)
#         idx = np.argsort(D)
#         pred = category_train[idx]
#         err += (compute_knn_score(k = 1, pred) == target)
#
#     # aggregate results
#     results = results + [err/len(docs_test)]

# test 1
#D = distance(reuters_test[:1], reuters_train[:1], 1, 100, data_path, verbose = 1)

# test 2
#D = distance(reuters_test[:1], reuters_train[:1000], 1, 100, data_path, verbose = 1)

# test 3
#D = distance(reuters_test[:1], reuters_train, 1, 100, data_path, verbose = 1)

# save results
#pkl.dump(D, open(data_path + "D_matrix.p", 'wb'))
