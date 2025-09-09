
from kfp import dsl
from kfp.dsl import Dataset, Model
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobRunOp

PROJECT_ID = "your-project-id"
REGION = "us-central1"
BUCKET = "gs://your-bucket/graphbert"
IMAGE_URI = "us-central1-docker.pkg.dev/my-gitrunning-55025/my-docker-repo/myapp:latest"  # 你打包的docker镜像

# 🔹 Step1: 数据准备
@dsl.component(base_image=IMAGE_URI)
def step1_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step1", "--workdir", workdir], check=True)

# 🔹 Step2: 子图 embedding
@dsl.component(base_image=IMAGE_URI)
def step2_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step2", "--workdir", workdir], check=True)

# 🔹 Step3: 配置
@dsl.component(base_image=IMAGE_URI)
def step3_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step3", "--workdir", workdir], check=True)

# 🔹 Step4: 预训练
@dsl.component(base_image=IMAGE_URI)
def pretrain_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "pretrain", "--workdir", workdir], check=True)

# 🔹 Step5: Fine-tuning
@dsl.component(base_image=IMAGE_URI)
def finetune_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "finetune", "--workdir", workdir], check=True)


# 🔹 组合成 Pipeline
@dsl.pipeline(
    name="graphbert-cpu-pipeline",
    pipeline_root=BUCKET,
)
def graphbert_pipeline(workdir: str = BUCKET):
    s1 = step1_op(workdir=workdir)
    s2 = step2_op(workdir=workdir).after(s1)
    s3 = step3_op(workdir=workdir).after(s2)
    s4 = pretrain_op(workdir=workdir).after(s3)
    s5 = finetune_op(workdir=workdir).after(s4)

