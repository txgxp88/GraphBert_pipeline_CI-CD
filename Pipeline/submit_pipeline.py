from google.cloud import aiplatform

aiplatform.init(project="my-gitrunning-55025", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="graphbert-pipeline-run",
    template_path="graphbert_pipeline.yaml",  # or .json
    pipeline_root="gs://my-graphbert-bucket/pipeline-root/",  # <-- Must be valid gs:// path
    parameter_values={
        "workdir": "gs://my-graphbert-bucket/graphbert"
    }
)

job.run()
