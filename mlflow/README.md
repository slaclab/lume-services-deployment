# MLFlow k8s stack

This file explains each of the deployments/services and state of things in this very early deployment of MLFlow

- **MLFlow server -** python slim container, runs commands to install mlflow and its reqirements. It contains arguments to enable authentification, connect to minio and posgress backends. Messy but functional
- **MinIO** - S3 compatible backend for storing blobs (unstructured and large files). In this context used to store model files, code files and data as well as anything that can be classified as an artifact.
- **MLFlow-DB** (postgres:15.0) - structured data store, stores experiment results (training curves, stats, etc script params).
- **Createbuckets** (MinIO Job) - Used for first time setup to create buckets for mlflow.


### TODO:

* [ ] Convert env-configmap.yml to a kubernates secret.
* [ ] Build a container rather than running the installation every time the container starts.
* [ ] Decide in default permission policy.
* [ ] Convert createbuckets job to something neater (Makefile might do the job?)
* [ ] Check this deployment against other deployments to conform to the same style.
* [ ] HTTPS, either though certificates or some https redirect (Kubernates has built in?)
* [ ] Once the above is done. Convert the old usage example from ISIS (Train and save model to MLFlow) script to work with SLAC deployment.
* [ ] Discuss where it would fit within the stack and plan for deployment and monitoring (Blue Sky at moment)
