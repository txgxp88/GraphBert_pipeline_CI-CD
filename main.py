from pathlib import Path
from step_1_processing import step_1
from step_2_subgraph import step_2
from step_3_setting import step_3
from step_4_classification import step_4

from Model.MethodGraphBertNode import MethodGraphBertNodeConstruct
from Model.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification, EvaluateAcc
from utilities.evaluation import evaluate_in_batches
import numpy as np

import torch
import torch.nn.functional as F
import time
import torch.optim as optim
import os

import argparse

def set_args(data):
    parser = argparse.ArgumentParser(description="Graph Model Training Settings")
    
    
    # Network setting
    parser.add_argument('--initializer_range', type=float, default=0.02, help='initializer_range')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number Of HiddenLayers')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention_heads')
    parser.add_argument('--intermediate_size', type=int, default=128, help='intermediate_size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2, help='drop out at hidden layer')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2, help='drop out at attentional layer')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='gelu activation function')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='layer_norm_eps')
    
    parser.add_argument('--residual_type', type=str, default='graph_raw', help='graph_raw or raw or None')
    
    #Bert Config
    parser.add_argument('--max_wl_role_index', type=int, default=100, help='max_wl_role_index')
    parser.add_argument('--max_hop_dis_index', type=int, default=100, help='max_hop_dis_index')
    parser.add_argument('--max_inti_pos_index', type=int, default=100, help='max_inti_pos_index')
    parser.add_argument('--top_k', type=int, default=7, help='top_k neighbors')
    parser.add_argument('--k', type=int, default=len(data.y.unique()), help='embedding dimension')
    
    #Data Config
    parser.add_argument('--nclass', type=int, default=len(data.y.unique()), help='nclass')
    parser.add_argument('--nfeature', type=int, default=data.x.shape[1], help='nfeature')
    parser.add_argument('--ngraph', type=int, default=data.x.shape[0], help='ngraph or nodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for the data split')
    
    
    # Early stop, optimizer etc. Config
    parser.add_argument('--patience', type=int, default=30, help='early stop patience')
    parser.add_argument('--mode', type=str, default='min', help='during the training the loss should be minimum')
    parser.add_argument('--base_lr', type=int, default=1e-3, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--factor', type=float, default=0.5, help='weight_decay')
    parser.add_argument('--decay_factor', type=float, default=0.9, help='decay_factor for lr modulation')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    
    # Localize current file path
    file_path = Path(__file__).resolve()

    dir_path = file_path.parent
    print(dir_path)
    

    #----------------------------------------------------------------------
    # 1, Data preparations which is a graph data structures
    #----------------------------------------------------------------------
    data = step_1(dir_path)
    args = set_args(data)
    
    
    #----------------------------------------------------------------------
    # 2, Data embedding
    #----------------------------------------------------------------------
    raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = step_2(dir_path, data, args.top_k)
    
    #----------------------------------------------------------------------
    # 3, Pretrained by the node perturbation
    #----------------------------------------------------------------------
    bert_config = step_3(data, args)
    
    GraphBertNodeConstruct = MethodGraphBertNodeConstruct(bert_config)

    max_epoch = 200

    node_learning_record_dict = {}

    t_begin = time.time()
    # optimizer = optim.AdamW(GraphBertNodeConstruct.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.AdamW(GraphBertNodeConstruct.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    print("Pretraining start!")
    for epoch in range(max_epoch):
        t_epoch_begin = time.time()

        # -------------------------

        GraphBertNodeConstruct.train()
        optimizer.zero_grad()

        output = GraphBertNodeConstruct.forward(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)

        # loss_train = F.mse_loss(output, data.x)
        loss_train = F.mse_loss(output, data.x)

        loss_train.backward()
        optimizer.step()

        node_learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}

        # -------------------------
        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'time: {:.4f}s'.format(time.time() - t_epoch_begin))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
    time.time() - t_begin
    
    #----------------------------------------------------------------------
    # 4, Fine-Tuning for the node classification
    #----------------------------------------------------------------------
    GraphBertNodeClassification = MethodGraphBertNodeClassification(bert_config)
    checkPoint_path = dir_path.joinpath('results','check_point')
    os.makedirs(os.path.dirname(f"{checkPoint_path}"), exist_ok=True)
    train_loader, test_loader, val_loader, accuracy, optimizer, scheduler, early_stopping = step_4(GraphBertNodeClassification, raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data, checkPoint_path, args)
    
    
    max_epoch = 100
    classify_learning_record_dict = {}
    max_score = 0.0
    t_begin = time.time()
    for epoch in range(max_epoch):
        t_epoch_begin = time.time()

        GraphBertNodeClassification.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for load in train_loader:

            optimizer.zero_grad()            
            
            output  = GraphBertNodeClassification.forward(
                raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, idx=load)
            
            # Two loss functions
            loss_train = F.cross_entropy(output, data.y[load])
            
            loss_train.backward()
            
            pred = (output).max(1)[1]
            correct = pred.eq(data.y[load]).sum().item()

            # epoch_loss += loss_train.item() * len(batch_idx)
            epoch_loss += loss_train.item() * len(load)# based on the number of samples
            
            optimizer.step()
            
            
            epoch_correct += correct
            
            # epoch_total += len(batch_idx)
            epoch_total += len(load)
            
            
            # print(f"Batch {i+1}/{num_batches}, Loss: {loss_train.item():.4f}, Accuracy: {correct/len(batch_idx):.4f}")
            

        loss_train_avg = epoch_loss / epoch_total
        acc_train_avg = epoch_correct / epoch_total

        # evaluation
        GraphBertNodeClassification.eval()#frozen
        
        # Validate (using mini-batch)
        acc_val, loss_val, accuracy_val, precision_val, recall_val, f1_val, roc_auc_val, confusion_matrix_val = evaluate_in_batches(GraphBertNodeClassification, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, val_loader, batch_size=20)

        # Test (using mini-batch)
        acc_test, loss_test, accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, confusion_matrix_test = evaluate_in_batches(GraphBertNodeClassification, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, test_loader, batch_size=20)

        classify_learning_record_dict[epoch] = {
            'loss_train': loss_train_avg,
            'acc_train': acc_train_avg,
            'loss_val': loss_val,
            'acc_val': acc_val,
            'loss_test': loss_test,
            'acc_test': acc_test,
            'time': time.time() - t_epoch_begin
        }

        print(f"Epoch: {epoch+1:04d}")
        print(f"  [Train] loss: {loss_train_avg:.4f} | acc: {acc_train_avg:.4f}")
        print(f"  [Valid] loss: {loss_val:.4f} | acc: {acc_val:.4f}")
        print(f"  [Test ] loss: {loss_test:.4f} | acc: {acc_test:.4f}")
        print(f"  Time: {time.time() - t_epoch_begin:.4f}s")
        print('-----------------------------------------')
        
        #lr scheduler
        scheduler.step(loss_val)
        

        early_stopping(loss_val, GraphBertNodeClassification)
        
        # Early stopping
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', \
        best testing performance {: 4f}'.format(np.max([classify_learning_record_dict[epoch]['acc_test'] for epoch in classify_learning_record_dict])) \
            + ', minimun loss {: 4f}'.format(np.min([classify_learning_record_dict[epoch]['loss_test'] for epoch in classify_learning_record_dict])))



    torch.save(classify_learning_record_dict, f'{checkPoint_path}/classify_learning_record_dict.pth')
    print(f"training procedure saved")
    
    torch.save(GraphBertNodeClassification.state_dict(), f'{checkPoint_path}/model_tuned.pt')
    print(f"Model saved")

    