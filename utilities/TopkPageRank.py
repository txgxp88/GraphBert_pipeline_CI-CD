import scipy.sparse as sp
import numpy as np

import networkx as nx
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


# building the adjacency matrix
def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # Sum over rows (degrees of nodes)
    r_inv = np.power(rowsum, -0.5).flatten()  # Inverse square root of degrees
    r_inv[np.isinf(r_inv)] = 0.  # Handle division by zero (e.g., for isolated nodes)
    r_mat_inv = sp.diags(r_inv)  # Create a sparse diagonal matrix from the inverse degrees
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)  # Apply the normalization formula
    return mx

# PageRank-inspired adjacent matrix for propogation
def neumann_approx_inverse(adj, c=0.15, K=10):
    """
    c = 0.15
    Neumann series approximation for eigen_adj = 0.15 * inv(I - (1 - 0.15) * A)
    A is the normalized adjacency matrix
    Returns a sparse matrix
    """
    alpha = 1 - c
    A = adj_normalize(adj)
    n = A.shape[0]

    # initial value is ones matrix
    result = sp.eye(n, format='csr', dtype=np.float32)
    A_power = sp.eye(n, format='csr', dtype=np.float32)  # A^0

    for k in range(1, K + 1):
        A_power = alpha * A.dot(A_power)  # A^k
        result += A_power

    return c * result  # c * sum(alpha^k A^k)


#Sparse version
def get_top_k_sparse(adj: sp.spmatrix, c: float, k: int):
    """
    eigen_adj: scipy.sparse.csr_matrix or coo_matrix
    k: int
    return: dict[node] = [(neighbor, score), ...]
    """
    
    # PageRank-inspired adjacent matrix for propogation
    eigen_adj = neumann_approx_inverse(adj, c=c, K=15)
    
    dense = eigen_adj.toarray()
    n = dense.shape[0]
    result_dict = {}
    
    key_map = None
    for i in range(n):
        scores = dense[i]
        scores[i] = -np.inf # remove self connection
        if k < n:
            top_k_idx = np.argpartition(-scores, k)[:k]
        else:
            top_k_idx = np.arange(n)
            top_k_idx = top_k_idx[top_k_idx != i]
        
        # sorting
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        neighbors = [(idx, scores[idx]) for idx in top_k_idx]
        mapped_node = key_map.get(i, i) if key_map else i
        mapped_neighbors = [(key_map.get(nid, nid) if key_map else nid, val) for nid, val in neighbors]

        result_dict[mapped_node] = mapped_neighbors

        

    return result_dict


# KBatch = get_top_k_sparse(adj, 0.15, 2)# Test



# BatchHopDistance typically refers to the shortest-path hop distance 
# computed between all pairs of nodes in batched graphs 
def BatchHopDistance(node_list, edge_index, k, batch_dict):
    import pickle
    import networkx as nx
    
    edge_index = edge_index.cpu().numpy()
    link_list = list(zip(edge_index[0], edge_index[1]))

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(link_list)

    hop_dict = {}

    for node in batch_dict:
        try:
            node_hop_lengths = nx.single_source_shortest_path_length(G, node, cutoff=10)
        except:
            node_hop_lengths = {}

        hop_dict[node] = {}
        for neighbor, _ in batch_dict[node]:
            hop = node_hop_lengths.get(neighbor, 99)
            hop_dict[node][neighbor] = hop

    return hop_dict
