from kfp import dsl
from kfp.dsl import Dataset, Model
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
import yaml
import kfp

# PROJECT_ID = "my-gitrunning-55025"
# REGION = "us-central1"
# BUCKET = "gs://my-graphbert-bucket/graphbert"
# IMAGE_URI = "us-central1-docker.pkg.dev/my-gitrunning-55025/my-docker-repo/myapp:latest"  # 你打包的docker镜像

with open("pipeline_config.yml", "r") as f:
    cfg = yaml.safe_load(f)


# 🔹 Step1: data preparation
@dsl.component(base_image=cfg["IMAGE_URI"])
def step1_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step1", "--workdir", workdir], check=True)

# 🔹 Step2: subgraph embedding
@dsl.component(base_image=cfg["IMAGE_URI"])
def step2_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step2", "--workdir", workdir], check=True)

# 🔹 Step3: setting
@dsl.component(base_image=cfg["IMAGE_URI"])
def step3_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "step3", "--workdir", workdir], check=True)

# 🔹 Step4: pretraining
@dsl.component(base_image=cfg["IMAGE_URI"])
def pretrain_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "pretrain", "--workdir", workdir], check=True)

# 🔹 Step5: Fine-tuning
@dsl.component(base_image=cfg["IMAGE_URI"])
def finetune_op(workdir: str):
    import subprocess
    subprocess.run(["python", "pipeline_runner.py", "--step", "finetune", "--workdir", workdir], check=True)


# 🔹 comprise Pipeline
@dsl.pipeline(
    name="graphbert-cpu-pipeline",
    pipeline_root=cfg["pipeline_root"],
)
def graphbert_pipeline(workdir: str):
    s1 = step1_op(workdir=workdir).set_caching_options(False)
    s2 = step2_op(workdir=workdir).after(s1).set_caching_options(False)
    s3 = step3_op(workdir=workdir).after(s2).set_caching_options(False)
    s4 = pretrain_op(workdir=workdir).after(s3).set_caching_options(False)
    s5 = finetune_op(workdir=workdir).after(s4).set_caching_options(False)

