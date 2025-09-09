import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import gcsfs

def step_1(dir_path: str):
    fs = gcsfs.GCSFileSystem()

    content_path = f"{dir_path}/Data/Cora/cora.content"
    cites_path = f"{dir_path}/Data/Cora/cora.cites"


    with fs.open(content_path, 'rb') as f:
        content_df = pd.read_csv(f, sep='\t', header=None)

    with fs.open(cites_path, 'rb') as f:
        cites_df = pd.read_csv(f, sep='\t', header=None, names=['source', 'target'])

    paper_ids = content_df[0].tolist()
    features = torch.tensor(content_df.iloc[:, 1:-1].values, dtype=torch.float)
    labels_raw = content_df.iloc[:, -1].tolist()

    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(labels_raw), dtype=torch.long)

    id_map = {pid: i for i, pid in enumerate(paper_ids)}

    cites_df = cites_df[cites_df['source'].isin(id_map) & cites_df['target'].isin(id_map)]
    src = cites_df['source'].map(id_map).tolist()
    dst = cites_df['target'].map(id_map).tolist()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=labels)

    num_classes = len(data.y.unique())
    y_one_hot = F.one_hot(data.y, num_classes=num_classes).float()

    data.num_nodes = data.x.shape[0]
    data.node_list = list(range(data.num_nodes))
    data.y_one_hot = y_one_hot

    return data
