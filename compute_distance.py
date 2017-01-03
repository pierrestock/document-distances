import numpy as np
import tensorflow as tf
import gensim as gs
import pickle as pkl

from time import time

def distance_gpu(D1, D2, gamma, niter, data_path, verbose = 0):
    """
    calculates all pairwise distances d(d1,d2) with:
        - d1 in D1 = [d11, d12, ...]
        - d2 in D2 = [d21, d22, ...]
        - gamma: entropic regularization parameter
        - niter: number of iterations of Sinkhorn's algorithm
        - data_path: path to data files
        - verbose = {0: display nothing, 1: display intermedidate computation times}
    using tensorflow on gpu
    """
    # load distance matrix and associated keys
    C = pkl.load(open(data_path + "C_most_common_5000.p", 'rb'))
    keys = pkl.load(open(data_path + "keys_most_common_5000.p", 'rb'))

    # define constants
    tm = time()
    D1_ready = []
    D2_ready = []
    support = keys.keys()
    n = len(support)
    d1 = len(D1)
    d2 = len(D2)

    # tokenize each document
    for doc in D1:
        D1_ready += [[w for w in gs.utils.tokenize(text=doc, lowercase=True) if w in support]]

    for doc in D2:
        D2_ready += [[w for w in gs.utils.tokenize(text=doc, lowercase=True) if w in support]]

    # count occurences
    P = np.zeros([d1, n])
    Q = np.zeros([d2, n])

    for i in range(d1):
        doc = D1_ready[i]
        for word in doc:
            P[i, keys[word]] += 1

    for i in range(d2):
        doc = D2_ready[i]
        for word in doc:
            Q[i, keys[word]] += 1

    # normalize and add default mass
    vmin = 0
    normalize = lambda P: P/np.tile(np.sum(P, 1, keepdims=True), n)
    P = normalize(P + np.max(P)*vmin)
    Q = normalize(Q + np.max(Q)*vmin)

    # stack the histograms to cover all possibilities
    P = np.repeat(P, d2, axis = 0)
    Q = np.vstack(d1 * [Q])

    if verbose == 1:
        print("Preprocessing: ", time() - tm )
    tm = time()

    # --- beginning of tf session --- #
    with tf.Session() as sess:

            #----------------- initialize graphs -----------------#
            # define constants
            P = tf.constant(P, dtype='float32')
            Q = tf.constant(Q, dtype='float32')
            C = tf.constant(C, dtype='float32')

            # define placeholders
            idx = tf.placeholder(dtype = 'int32', shape = None)

            # define variables
            xi = tf.Variable(C, dtype = 'float32')
            A = tf.Variable(tf.ones([d1 * d2, n], dtype='float32'), dtype='float32')
            B = tf.Variable(tf.ones([d1 * d2, n], dtype='float32'), dtype='float32')

            # define graphs
            init = tf.initialize_all_variables()

            op_xi = tf.exp(tf.scalar_mul(-1./gamma,tf.pow(xi, 2)))
            update_xi = tf.assign(xi, op_xi)

            op_A = tf.div(P, tf.matmul(B, xi))
            update_A = tf.assign(A, op_A)
            op_B = tf.div(Q, tf.matmul(A, xi))
            update_B = tf.assign(B, op_B)

            A_row_as_col = tf.reshape(tf.slice(A, [idx, 0], [1, -1]), [-1, 1])
            B_row_as_row = tf.reshape(tf.slice(B, [idx, 0], [1, -1]), [1, -1])
            compute_transp = tf.mul(A_row_as_col, tf.mul(xi, B_row_as_row)) # note that tf.mul supports broadcasting
            compute_D = tf.reduce_sum(tf.mul(C, compute_transp))

            #-------------------- run graphs --------------------#
            # initialize variables
            sess.run(init)

            if verbose == 1:
                print("Initialize graphs/variables: ", time() - tm )
            tm = time()

            # compute xi
            sess.run(update_xi)

            if verbose == 1:
                print("Cost matrix: ", time() - tm )
            tm = time()

            # Sinkhorn's alrogithm
            for _ in range(niter):
                sess.run(update_A)
                sess.run(update_B)

            if verbose == 1:
                print("Sinkhorn: ", time() - tm )
            tm = time()

            # compute distance between P[i] and Q[i] and store result
            D = []
            for i in range(d1 * d2):
                d_current = sess.run(compute_D, feed_dict = {idx:i})
                D.append(d_current)

            if verbose == 1:
                print("Distance matrix: ", time() - tm )
            tm = time()

    # --- end of tf session --- #

    return np.reshape(D, [d1, d2])