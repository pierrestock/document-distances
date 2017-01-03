import numpy as np
import pickle as pkl
import gensim as gs
import os.path
from numpy import linalg

def cost_matrix(size_list, p_list, data_path):

    # load the word2vec model
    print("Loading word2vec model...")
    w2v_model = gs.models.Word2Vec.load_word2vec_format(data_path + "GoogleNews-vectors-negative300.bin.gz", binary=True)
    print("Done!")

    # load stopwords
    with open(data_path + "stop_words.txt") as f:
        content = f.read()
        stop_words = content.split()

    # load the 10,000 most common words in english
    # remove stopwords and test if they belong to the model
    most_common_words = []
    with open(data_path + "google-10000-english.txt") as f:
        for word in f:
            word = word[:-1]
            if word not in stop_words:
                delete = False
                try:
                    w2v_model[word]
                except KeyError:
                    delete = True
                if not delete:
                    most_common_words = most_common_words + [word]

    # compute keys and cost matrix
    C_most_common_names, keys_names = [], []

    for size in size_list:
        for p in p_list:
            print("Setting size = %d, p = %d" %(size, p))

            # if matrix already exists, do nothing
            C_current_name = "C_most_common_" + str(size) + "_" + str(p) + ".p"
            keys_current_name = "keys_most_common_" + str(size) + "_" + str(p) + ".p"
            if not(os.path.exists(C_current_name) and os.path.exists(keys_current_name)):

                # check if we have enough words
                assert(len(most_common_words) >= size)
                most_common_words_current = most_common_words[:size]

                C_most_common = np.zeros([size, size])
                keys = dict((e[1], e[0]) for e in enumerate(most_common_words_current))

                for i in range(size):
                    for j in range(i + 1, size):
                        C_most_common[i, j] = linalg.norm(w2v_model[most_common_words_current[i]] - w2v_model[most_common_words_current[j]], ord = p)

                # make C symmetric
                C_most_common = C_most_common + np.transpose(C_most_common)

                # save keys and cost matrix
                C_most_common_names += [C_current_name]
                keys_names += [keys_current_name]
                pkl.dump(C_most_common, open(data_path + C_current_name, 'wb'))
                pkl.dump(keys, open(data_path + keys_current_name, 'wb'))

    # return names
    return C_most_common_names, keys_names
