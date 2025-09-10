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
import yaml

# ------------------- save/read from GCS -------------------
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


def set_args(data, yaml_path=None):
    import argparse
    parser = argparse.ArgumentParser(description="Graph Model Training Settings")

    if yaml_path:
        if yaml_path.startswith("gs://"):
            client = storage.Client()
            bucket_name, blob_name = yaml_path[5:].split("/", 1)
            tmp_path = "/tmp/model_training_config.yml"
            print(f"[set_args] Downloading {yaml_path} -> {tmp_path}")
            client.bucket(bucket_name).blob(blob_name).download_to_filename(tmp_path)
            yaml_path = tmp_path

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    
    # 用 yaml 里的参数覆盖默认值
    parser.add_argument('--initializer_range', type=float, default=cfg.get("initializer_range", 0.02))
    parser.add_argument('--num_hidden_layers', type=int, default=cfg.get("num_hidden_layers", 2))
    parser.add_argument('--hidden_size', type=int, default=cfg.get("hidden_size", 32))
    # ... 其他参数照样加
    
    # 数据相关还是要从 data 里动态获取
    parser.add_argument('--k', type=int, default=len(data.y.unique()))
    parser.add_argument('--nclass', type=int, default=len(data.y.unique()))
    parser.add_argument('--nfeature', type=int, default=data.x.shape[1])
    parser.add_argument('--ngraph', type=int, default=data.x.shape[0])

    args = parser.parse_args([])
    return args


# ------------------- main -------------------
def main(step, workdir):
    os.makedirs("/tmp/workdir", exist_ok=True)  # local temp dir for intermediate files

    if step == "step1":
        data = step_1(workdir)  # workdir is bucket gs:// path
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
        args = set_args(data, f"{workdir}/model_training_config.yml")
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
        
        print("=== ENTER FINETUNE ===", flush=True)

        # ------------------- load data & config -------------------
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = load_obj(f"{workdir}/embeddings.pth")
        bert_config, args = load_obj(f"{workdir}/bert_config.pth")
        print("Loaded embeddings and config OK", flush=True)

        bert_config.output_attentions = False
        bert_config.output_hidden_states = False
        print("Changed config flags", flush=True)

        # ------------------- device -------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device, flush=True)

        GraphBertNodeClassification = MethodGraphBertNodeClassification(bert_config).to(device)
        raw_embeddings = raw_embeddings.to(device)
        wl_embedding = wl_embedding.to(device)
        hop_embeddings = hop_embeddings.to(device)
        int_embeddings = int_embeddings.to(device)
        data = data.to(device)
        print("Model and data moved to device", flush=True)

        # ------------------- checkpoint path -------------------
        local_checkpoint_path = "/tmp/check_point"
        os.makedirs(local_checkpoint_path, exist_ok=True)
        print("Checkpoint path prepared:", local_checkpoint_path, flush=True)

        # ------------------- loaders, optimizer, scheduler -------------------
        train_loader, test_loader, val_loader, accuracy, optimizer, scheduler, early_stopping = step_4(
            GraphBertNodeClassification,
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data,
            local_checkpoint_path, args
        )
        print("Step 4 setup OK", flush=True)

        # ------------------- training loop -------------------
        max_epoch = 100
        classify_learning_record_dict = {}
        t_begin = time.time()

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            GraphBertNodeClassification.train()

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for load in train_loader:
                optimizer.zero_grad()
                idx = load
                output = GraphBertNodeClassification(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, idx=idx)
                loss_train = F.cross_entropy(output, data.y[idx])
                loss_train.backward()
                optimizer.step()

                pred = output.max(1)[1]
                correct = pred.eq(data.y[idx]).sum().item()

                epoch_loss += loss_train.item() * len(idx)
                epoch_correct += correct
                epoch_total += len(idx)

            loss_train_avg = epoch_loss / epoch_total
            acc_train_avg = epoch_correct / epoch_total

            # ------------------- evaluation -------------------
            GraphBertNodeClassification.eval()
            acc_val, loss_val, accuracy_val, precision_val, recall_val, f1_val, roc_auc_val, confusion_matrix_val = evaluate_in_batches(
                GraphBertNodeClassification, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, val_loader, batch_size=20
            )
            acc_test, loss_test, accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, confusion_matrix_test = evaluate_in_batches(
                GraphBertNodeClassification, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, test_loader, batch_size=20
            )

            classify_learning_record_dict[epoch] = {
                'loss_train': loss_train_avg,
                'acc_train': acc_train_avg,
                'loss_val': loss_val,
                'acc_val': acc_val,
                'loss_test': loss_test,
                'acc_test': acc_test,
                'time': time.time() - t_epoch_begin
            }

            print(f"Epoch {epoch+1:04d} | Train loss {loss_train_avg:.4f}, acc {acc_train_avg:.4f} | "
                f"Val loss {loss_val:.4f}, acc {acc_val:.4f} | Test acc {acc_test:.4f} | "
                f"Time {time.time()-t_epoch_begin:.2f}s", flush=True)

            scheduler.step(loss_val)
            early_stopping(loss_val, GraphBertNodeClassification)

            if early_stopping.early_stop:
                print("Early stopping triggered.", flush=True)
                break

        print("Optimization Finished!", flush=True)
        print("Total time elapsed: {:.2f}s".format(time.time()-t_begin), flush=True)

        # ------------------- save results -------------------
        save_obj(classify_learning_record_dict, f"{workdir}/classify_learning_record_dict.pth")
        save_obj(GraphBertNodeClassification.state_dict(), f"{workdir}/model_tuned.pt")
        print("Training records and model saved", flush=True)


    else:
        raise ValueError(f"Unknown step {step}")

# ------------------- CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    args = parser.parse_args()
    main(args.step, args.workdir)
