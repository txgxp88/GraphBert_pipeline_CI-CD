# The GCP pipeline in the example of Basic version of Graph-Bert Model



----------------------------------------------------------
## The procedure of our code?

According to Graph-Bert Model, there are 4 steps:

(1) Data Fetching and Processing: In this step, we use a simple example Cora dataset to demonstrate the entire pipeline

(2) Graph-Bert Model Input preparation with data embedding: (a) Node WL Code. (b) Intimacy-based Subgraph Batch. (c) Node Hop Distance

(3) Graph-Bert parameters setting

(4) Pretraining

(5) Fine-tuning tasks for node classification 

<img width="1436" height="1194" alt="image" src="https://github.com/user-attachments/assets/0ecb83aa-3228-4a70-b34b-aa04ef7b9e04" />


----------------------------------------------------------
## How to use on GCP cloud

Folder descriptions

1. Bucket Folder (GCS)
   
   Stores intermediate data and hyperparameter config YAML file and GCS acts as persistent storage for passing data between steps.

2. Shell Pipeline Folder (local / Git repo)

   stores Vertex AI pipeline definition YAMLs (These YAMLs are only for orchestration, not for storing data.)

3. Container Image

   Packages the code and its dependencies
