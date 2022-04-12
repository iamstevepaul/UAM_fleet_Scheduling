import time
import tensorflow as tf
from utils import *
import random
from topology import *
import igraph as ig
from igraph import ADJ_MAX
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import networkx as nx
from sklearn import preprocessing
import operator
import matplotlib.pyplot as plt
import ot

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)
adj_array = adj.toarray().astype(np.float32)
laplacian = preprocess_untuple_adj(adj, -0.5).toarray()

#var preprocess
def var_preprocess(adj, r):
    adj_ = adj + sp.eye(adj.shape[0])
    adj_ = adj_ ** r
    adj_[adj_ > 1] = 1
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


var_laplacian = var_preprocess(adj= adj, r= 2).toarray()
node_similarity_mat = np.load('node_similarity_mat.npy')
secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat = adj_array, w_adj_mat= node_similarity_mat, k =2)


reg_dgms = list()
for i in range(len(secondorder_subgraph)):
    print(i)
    tmp_reg_dgms = simplicial_complex_dgm(secondorder_subgraph[i])
    if tmp_reg_dgms.size == 0:
        reg_dgms.append(np.array([]))
    else:
        reg_dgms.append(np.unique(tmp_reg_dgms, axis=0))

reg_dgms = np.array(reg_dgms)
np.savez("reg_dgms", reg_dgms)


row_labels = np.where(var_laplacian>0.)[0]
col_labels = np.where(var_laplacian>0.)[1]

topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype = np.float32)

for i in range(row_labels.shape[0]):
    tmp_row_label = row_labels[i]
    tmp_col_label = col_labels[i]
    tmp_wasserstin_dis = wasserstein(reg_dgms[tmp_row_label], reg_dgms[tmp_col_label])
    if tmp_wasserstin_dis == 0.:
        topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1./1e-5
        topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1./1e-5
    else:
        topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / tmp_wasserstin_dis
        topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / tmp_wasserstin_dis

np.save('inverse_wasserstein_distance',topo_laplacian_k_2)

# Based on Hamming distance
row_labels = np.where(var_laplacian>0.)[0]
col_labels = np.where(var_laplacian>0.)[1]

topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype = np.float32)
second_order_subgraph_features = k_th_order_subgraph_features(adj_mat = adj_array, features = features.toarray(), k =2)

for i in range(row_labels.shape[0]):
    tmp_row_label = row_labels[i]
    tmp_col_label = col_labels[i]
    tmp_wasserstin_dis = ot.emd2([], [], ot.dist(second_order_subgraph_features[tmp_row_label],second_order_subgraph_features[tmp_col_label]))

    if tmp_wasserstin_dis == 0.:
        topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1./1e-5
        topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1./1e-5
    else:
        topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / tmp_wasserstin_dis
        topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / tmp_wasserstin_dis

np.save('inverse_hamming_distance',topo_laplacian_k_2)
