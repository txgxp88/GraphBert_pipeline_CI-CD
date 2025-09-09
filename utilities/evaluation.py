import torch
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ------------------------------------
# evaluation function
def evaluate_in_batches(model, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, idx_eval, batch_size=10):
    model.eval()

    with torch.no_grad():
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for idx in idx_eval:
            
            
            label_ouputs = model.forward(raw_embeddings, wl_embedding,
                                int_embeddings, hop_embeddings, data, idx)
            
                
            loss_train_cls_residual = F.cross_entropy(label_ouputs, data.y[idx])
        
            epoch_loss += loss_train_cls_residual.item() * len(data.y[idx])# based on the number of samples
            
            
            #### examine the metrics
            probs = F.softmax(label_ouputs, dim=1)  # shape: (batch, 2)
            preds = torch.argmax(label_ouputs, dim=1)  # predicted class
            
            
            all_probs.append(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(data.y[idx].numpy())
            
            
            epoch_correct += preds.eq(data.y[idx]).sum().item()
            epoch_total += len(label_ouputs)
            
        
        
            
        loss_train_avg = epoch_loss / epoch_total
        acc_train_avg = epoch_correct / epoch_total
        
        # Calculate AUC
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        # all_probs = np.array(all_probs)
        all_probs = np.concatenate(all_probs, axis=0)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class='ovr')
        
        
        cm = confusion_matrix(all_labels, all_preds).ravel()
            
    return acc_train_avg, loss_train_avg, accuracy, precision, recall, f1, roc_auc, cm

# ------------------------------------
# early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.path = path

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = None

        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, val_metric, model, epoch=None):
        score = val_metric if self.mode == 'min' else -val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model, val_metric)
        elif score < self.best_score - self.delta:  # ✅ only when truly improved
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model, val_metric)
        else:
            self.counter += 1
            print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, val_metric):
        torch.save(model.state_dict(), self.path)
        print(f"Model improved. Saved to {self.path} | Val Metric: {val_metric:.4f}")



