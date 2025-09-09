import torch
import torch.nn.functional as F
import scipy.sparse as sp
import os
from numpy.linalg import inv
import numpy as np
import pickle

from utilities.WLGraphColoring import WLGraphColoring
from utilities.TopkPageRank import get_top_k_sparse, BatchHopDistance


# building the adjacency matrix
def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # Sum over rows (degrees of nodes)
    r_inv = np.power(rowsum, -0.5).flatten()  # Inverse square root of degrees
    r_inv[np.isinf(r_inv)] = 0.  # Handle division by zero (e.g., for isolated nodes)
    r_mat_inv = sp.diags(r_inv)  # Create a sparse diagonal matrix from the inverse degrees
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)  # Apply the normalization formula
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def step_2(dir_path, data, k):
    ########## one-hot encoding and similarity matrix calculations
    num_classes = data.num_nodes # Cora has 7 classes

    # F.one_hot for one-hot coding
    y_one_hot = data.y_one_hot  # cora dataset shape: [2708, 7]

    edges_index = data.edge_index.cpu().numpy().T  # [num_edges, 2]
    adj = sp.coo_matrix(
        (np.ones(edges_index.shape[0]), (edges_index[:, 0], edges_index[:, 1])),
        shape=(y_one_hot.shape[0], y_one_hot.shape[0]),
        dtype=np.float32
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))# normalized adjacency matrix

    data.adj = sparse_mx_to_torch_sparse_tensor(norm_adj)


    ########## Weisfeiler-Lehman (WL) based graph coloring
    wl_coloring = WLGraphColoring()

    # Assuming `data.node_list` and `data.edge_index` are your node list and edge index
    wl_coloring.setting_init(data.node_list, data.edge_index)
    wl_coloring.WL_recursion(data.node_list)

    # Save the node color dict
    saving_path = dir_path.joinpath('results')
    os.makedirs(os.path.dirname(f"{saving_path}/WL/WL"), exist_ok=True)
    wl_coloring.save_coloring(f'{saving_path}/WL/WL')
    # wl_coloring.visualize_graph()# Visualize the graph coloring if it necessary
    f = open(f'{saving_path}/WL/WL', 'rb')
    wl_dict = pickle.load(f)
    f.close()



    ########## Top-k Personalized PageRank neighbor for propagation
    batch_dict = get_top_k_sparse(adj, 0.15, k)
    #saving files
    os.makedirs(os.path.dirname(f"{saving_path}/Batch/top_{k}_GraphBatching"), exist_ok=True)
    f = open(f"{saving_path}/Batch/top_{k}_GraphBatching", 'wb')
    pickle.dump(batch_dict, f)
    f.close()


    ########## Top-k Personalized Hop distance for propagation
    hop_dict = BatchHopDistance(data.node_list, data.edge_index, k, batch_dict)
    os.makedirs(os.path.dirname(f"{saving_path}/Hop/top_{k}_GraphBatchingHop"), exist_ok=True)
    f = open(f"{saving_path}/Hop/top_{k}_GraphBatchingHop", 'wb')
    pickle.dump(hop_dict, f)
    f.close()



    ###### ---------- embedding process
    raw_feature_list = []
    role_ids_list = []
    position_ids_list = []
    hop_ids_list = []
    idx = data.node_list

    for node in idx:
        node_index = node 
        neighbors_list = batch_dict[node]

        raw_feature = [data.x[node_index].tolist()]
        role_ids = [wl_dict[node]]
        position_ids = range(len(neighbors_list) + 1)
        hop_ids = [0]
        for neighbor, intimacy_score in neighbors_list:
            neighbor_index = neighbor
            
            raw_feature.append(data.x[neighbor_index].tolist())
            role_ids.append(wl_dict[neighbor])
            if neighbor in hop_dict[node]:
                hop_ids.append(hop_dict[node][neighbor])
            else:
                hop_ids.append(99)
        raw_feature_list.append(raw_feature)
        role_ids_list.append(role_ids)
        position_ids_list.append(position_ids)
        hop_ids_list.append(hop_ids)

    raw_embeddings = torch.FloatTensor(raw_feature_list)
    wl_embedding = torch.LongTensor(role_ids_list)
    hop_embeddings = torch.LongTensor(hop_ids_list)
    int_embeddings = torch.LongTensor(position_ids_list)
    
    return raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data








