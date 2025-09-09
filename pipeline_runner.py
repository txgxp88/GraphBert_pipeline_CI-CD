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

from google.cloud import storage

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
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        tmp_path = "/tmp/tmp_load.pth"
        client.bucket(bucket_name).blob(blob_name).download_to_filename(tmp_path)
        return torch.load(tmp_path)
    else:
        return torch.load(path)

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
        from main import set_args
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

    elif step == "finetune":
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data = load_obj(f"{workdir}/embeddings.pth")
        bert_config, args = load_obj(f"{workdir}/bert_config.pth")

        GraphBertNodeClassification = MethodGraphBertNodeClassification(bert_config)
        checkPoint_path = f"{workdir}/check_point"
        os.makedirs("/tmp/check_point", exist_ok=True)

        train_loader, test_loader, val_loader, accuracy, optimizer, scheduler, early_stopping = step_4(
            GraphBertNodeClassification, raw_embeddings, wl_embedding, hop_embeddings, int_embeddings, data, checkPoint_path, args
        )

        max_epoch = 100
        classify_learning_record_dict = {}
        for epoch in range(max_epoch):
            GraphBertNodeClassification.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            for load in train_loader:
                optimizer.zero_grad()
                output = GraphBertNodeClassification(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings, data, idx=load)
                loss_train = F.cross_entropy(output, data.y[load])
                loss_train.backward()
                pred = output.max(1)[1]
                correct = pred.eq(data.y[load]).sum().item()
                optimizer.step()
                epoch_loss += loss_train.item() * len(load)
                epoch_correct += correct
                epoch_total += len(load)

            print(f"[Finetune] epoch {epoch}, loss={epoch_loss/epoch_total:.4f}, acc={epoch_correct/epoch_total:.4f}")

            scheduler.step(epoch_loss)
            early_stopping(epoch_loss, GraphBertNodeClassification)
            if early_stopping.early_stop:
                break

        save_obj(classify_learning_record_dict, f"{workdir}/classify_learning_record_dict.pth")
        save_obj(GraphBertNodeClassification.state_dict(), f"{workdir}/model_tuned.pt")

    else:
        raise ValueError(f"Unknown step {step}")

# ------------------- CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    args = parser.parse_args()
    main(args.step, args.workdir)
