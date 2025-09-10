import torch
import torch.nn.functional as F
import scipy.sparse as sp
import os
import pickle
import tempfile
from utilities.WLGraphColoring import WLGraphColoring
from utilities.TopkPageRank import get_top_k_sparse, BatchHopDistance
import numpy as np

from google.cloud import storage

def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def upload_to_gcs(local_path, gcs_path):
    #upload_to_gcs(local_path, gcs_path):
    if gcs_path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = gcs_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_filename(local_path)

def step_2(dir_path, data, top_k):
    ########## adjacency
    y_one_hot = data.y_one_hot
    edges_index = data.edge_index.cpu().numpy().T
    adj = sp.coo_matrix(
        (np.ones(edges_index.shape[0]), (edges_index[:, 0], edges_index[:, 1])),
        shape=(y_one_hot.shape[0], y_one_hot.shape[0]),
        dtype=np.float32
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))
    data.adj = sparse_mx_to_torch_sparse_tensor(norm_adj)

    ########## WL coloring
    wl_coloring = WLGraphColoring()
    wl_coloring.setting_init(data.node_list, data.edge_index)
    wl_coloring.WL_recursion(data.node_list)

    # temporal store WL coloring
    tmp_dir = tempfile.mkdtemp()
    wl_local_path = os.path.join(tmp_dir, "WL.pkl")
    wl_coloring.save_coloring(wl_local_path)
    wl_dict = pickle.load(open(wl_local_path, "rb"))

    # upload to GCS
    wl_gcs_path = os.path.join(dir_path, "WL/WL.pkl")
    upload_to_gcs(wl_local_path, wl_gcs_path)

    ########## Top-k Personalized PageRank
    batch_dict = get_top_k_sparse(adj, 0.15, top_k)
    batch_local_path = os.path.join(tmp_dir, f"top_{top_k}_GraphBatching.pkl")
    with open(batch_local_path, "wb") as f:
        pickle.dump(batch_dict, f)
    batch_gcs_path = os.path.join(dir_path, f"Batch/top_{top_k}_GraphBatching.pkl")
    upload_to_gcs(batch_local_path, batch_gcs_path)

    ########## Top-k Hop distance
    hop_dict = BatchHopDistance(data.node_list, data.edge_index, top_k, batch_dict)
    hop_local_path = os.path.join(tmp_dir, f"top_{top_k}_GraphBatchingHop.pkl")
    with open(hop_local_path, "wb") as f:
        pickle.dump(hop_dict, f)
    hop_gcs_path = os.path.join(dir_path, f"Hop/top_{top_k}_GraphBatchingHop.pkl")
    upload_to_gcs(hop_local_path, hop_gcs_path)

    ###### embedding process
    raw_feature_list, role_ids_list, position_ids_list, hop_ids_list = [], [], [], []
    for node in data.node_list:
        neighbors_list = batch_dict[node]
        raw_feature = [data.x[node].tolist()]
        role_ids = [wl_dict[node]]
        position_ids = list(range(len(neighbors_list) + 1))
        hop_ids = [0]
        for neighbor, _ in neighbors_list:
            raw_feature.append(data.x[neighbor].tolist())
            role_ids.append(wl_dict[neighbor])
            hop_ids.append(hop_dict[node].get(neighbor, 99))
        raw_feature_list.append(raw_feature)
        role_ids_list.append(role_ids)
        position_ids_list.append(position_ids)
        hop_ids_list.append(hop_ids)

    raw_embeddings = torch.FloatTensor(raw_feature_list)
    wl_embedding = torch.LongTensor(role_ids_list)
    hop_embeddings = torch.LongTensor(hop_ids_list)
    int_embeddings = torch.LongTensor(position_ids_list)

    return raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data
