from kfp import compiler
from graphbert_vertex_pipeline import graphbert_pipeline  # your pipeline function

compiler.Compiler().compile(
    pipeline_func=graphbert_pipeline,
    package_path='graphbert_pipeline.yaml'  # YAML is preferred format
)
