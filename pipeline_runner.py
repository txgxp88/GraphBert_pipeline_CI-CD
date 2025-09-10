import argparse
import torch
import os
from step_1_processing import step_1
from step_2_subgraph import step_2
from step_3_setting import step_3
from step_4_classification import step_4
from Model.MethodGraphBertNode import MethodGraphBertNodeConstruct
from Model.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from utilities.evaluation import evaluate_in_batches
import torch.nn.functional as F
import torch.optim as optim
import gcsfs
import argparse
    
from google.cloud import storage
import os
print("GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
import time
import numpy as np

# ------------------- 保存/读取支持 GCS -------------------
def save_obj(obj, path):
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        tmp_path = "/tmp/tmp_save.pth"
        torch.save(obj, tmp_path)
        bucket.blob(blob_name).upload_from_filename(tmp_path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)

def load_obj(path):
    try:
        if path.startswith("gs://"):
            client = storage.Client()
            bucket_name, blob_name = path[5:].split("/", 1)
            tmp_path = "/tmp/tmp_load.pth"
            print(f"[load_obj] Downloading gs://{bucket_name}/{blob_name} -> {tmp_path}")
            blob = client.bucket(bucket_name).blob(blob_name)
            if not blob.exists():
                raise FileNotFoundError(f"Blob gs://{bucket_name}/{blob_name} does not exist")
            blob.download_to_filename(tmp_path)
            print(f"[load_obj] Download complete, loading with torch.load()")
            obj = torch.load(tmp_path)
            print(f"[load_obj] Load successful")
            return obj
        else:
            print(f"[load_obj] Loading local file {path}")
            obj = torch.load(path)
            print(f"[load_obj] Load successful")
            return obj
    except Exception as e:
        print(f"[load_obj] Failed to load object from {path}: {e}")
        raise


def set_args(data):
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Model Training Settings")
    
    # Network setting
    parser.add_argument('--initializer_range', type=float, default=0.02)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--residual_type', type=str, default='graph_raw')
    
    # Bert Config
    parser.add_argument('--max_wl_role_index', type=int, default=100)
    parser.add_argument('--max_hop_dis_index', type=int, default=100)
    parser.add_argument('--max_inti_pos_index', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=7)
    parser.add_argument('--k', type=int, default=len(data.y.unique()))
    
    # Data Config
    parser.add_argument('--nclass', type=int, default=len(data.y.unique()))
    parser.add_argument('--nfeature', type=int, default=data.x.shape[1])
    parser.add_argument('--ngraph', type=int, default=data.x.shape[0])
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Training Config
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--mode', type=str, default='min')
    parser.add_argument('--base_lr', type=float, default=1e-3)   # 改成 float
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--decay_factor', type=float, default=0.9)
    
    # 不从 sys.argv 解析，而是直接取默认值
    args = parser.parse_args([])
    return args


# ------------------- 主流程 -------------------
def main(step, workdir):
    os.makedirs("/tmp/workdir", exist_ok=True)  # 本地缓存文件夹

    if step == "step1":
        data = step_1(workdir)  # workdir 是 gs:// 路径
        save_obj(data, f"{workdir}/data.pth")
        print("workdir:", workdir)
        print("data:", data)


    elif step == "step2":
        print(f"{workdir}/data.pth")

        data = load_obj(f"{workdir}/data.pth")
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = step_2(workdir, data, top_k=7)
        save_obj((raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data), f"{workdir}/embeddings.pth")

    elif step == "step3":
        _, _, _, _, data = load_obj(f"{workdir}/embeddings.pth")        
        args = set_args(data)
        bert_config = step_3(data, args)
        save_obj((bert_config, args), f"{workdir}/bert_config.pth")

    elif step == "pretrain":
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = load_obj(f"{workdir}/embeddings.pth")
        bert_config, args = load_obj(f"{workdir}/bert_config.pth")

        GraphBertNodeConstruct = MethodGraphBertNodeConstruct(bert_config)
        optimizer = optim.AdamW(GraphBertNodeConstruct.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

        max_epoch = 200
        for epoch in range(max_epoch):
            GraphBertNodeConstruct.train()
            optimizer.zero_grad()
            output = GraphBertNodeConstruct(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)
            loss_train = F.mse_loss(output, data.x)
            loss_train.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"[Pretrain] epoch {epoch}, loss={loss_train.item():.4f}")

        save_obj(GraphBertNodeConstruct.state_dict(), f"{workdir}/pretrained_model.pth")
        print(f"[Pretrain is done")
        
    elif step == "finetune":
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = load_obj(f"{workdir}/embeddings.pth")
        bert_config, args = load_obj(f"{workdir}/bert_config.pth")
        bert_config.output_attentions = False
        bert_config.output_hidden_states = False
        print("change successful")
        
        GraphBertNodeClassification = MethodGraphBertNodeClassification(bert_config)
        checkPoint_path = f"{workdir}/check_point"
        os.makedirs("/tmp/check_point", exist_ok=True)

        train_loader, test_loader, val_loader, accuracy, optimizer, scheduler, early_stopping = step_4(
            GraphBertNodeClassification, raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data, checkPoint_path, args
        )
        
        
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
                
                output  = GraphBertNodeClassification(
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


        save_obj(classify_learning_record_dict, f"{workdir}/classify_learning_record_dict.pth")
        print(f"training procedure saved")
        save_obj(GraphBertNodeClassification.state_dict(), f"{workdir}/model_tuned.pt")
        print(f"Model saved")

    else:
        raise ValueError(f"Unknown step {step}")

# ------------------- CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    args = parser.parse_args()
    main(args.step, args.workdir)
