import time
from Model.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification, EvaluateAcc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from utilities.evaluation import evaluate_in_batches, EarlyStopping
import numpy as np



def step_4(GraphBertNodeClassification, raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data, checkPoint_path, args):

    # Separate dataset for purpose
    idx_train = range(140)
    idx_test = range(200, 1200)
    idx_val = range(1200, 1500)

    # train_loader = DataLoader(idx_train, batch_size=64, shuffle=True)
    # test_loader = DataLoader(idx_test, batch_size=64, shuffle=True)
    # val_loader = DataLoader(idx_val, batch_size=64, shuffle=True)
    
    train_loader = DataLoader(idx_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(idx_test, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(idx_val, batch_size=args.batch_size, shuffle=True)


    # ------------------------------------
    params = []
    base_lr = args.base_lr#1e-3
    decay_factor = args.decay_factor#0.9
    
    for i, (name, param) in enumerate(GraphBertNodeClassification.named_parameters()):
        lr = base_lr * (decay_factor ** (len(list(GraphBertNodeClassification.named_children())) - i - 1))
        params.append({'params': param, 'lr': lr})
        
    # ------------------------------------

    t_begin = time.time()

    optimizer = optim.Adam(params, lr=base_lr, weight_decay=1e-4)

    accuracy = EvaluateAcc('', '')

    # initialization 
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True)



    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, mode=args.mode, path=f'{checkPoint_path} \
                                checkpoint.pt')
    # early_stopping = EarlyStopping(patience=30, mode='min', path=f'{checkPoint_path} \
    #                             checkpoint.pt')


    return train_loader, test_loader, val_loader, accuracy, optimizer, scheduler, early_stopping