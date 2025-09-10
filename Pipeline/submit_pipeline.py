import yaml
from google.cloud import aiplatform

with open("pipeline_config.yml", "r") as f:
    cfg = yaml.safe_load(f)

# initialize the 
aiplatform.init(project=cfg["project"], location=cfg["location"])

# create and run the pipeline job
job = aiplatform.PipelineJob(
    display_name=cfg["display_name"],
    template_path=cfg["template_path"],
    pipeline_root=cfg["pipeline_root"],
    parameter_values=cfg["parameter_values"]
)

job.run()


