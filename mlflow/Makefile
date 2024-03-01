NAMESPACE ?= mlflow
operator:
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/dev-mlflow-branch/mlflow/createbuckets-deployment.yaml > createbuckets-deployment.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/dev-mlflow-branch/mlflow/env-configmap.yaml > env-configmap.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/minio-claim0-persistentvolumeclaim.yaml > minio-claim0-persistentvolumeclaim.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/minio-deployment.yaml > minio-deployment.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/minio-service.yaml > minio-service.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-db-claim0-persistentvolumeclaim.yaml > mlflow-db-claim0-persistentvolumeclaim.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-server-claim0-persistentvolumeclaim.yaml > mlflow-server-claim0-persistentvolumeclaim.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-db-deployment.yaml > mlflow-db-deployment.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-db-service.yaml > mlflow-db-service.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-server-deployment.yaml > mlflow-server-deployment.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/mlflow-server-service.yaml > mlflow-server-service.yaml
	curl https://raw.githubusercontent.com/slaclab/lume-services-deployment/raw/dev-mlflow-branch/mlflow/kustomization.yaml > kustomization.yaml
apply-secrets:
	kubectl create secret generic --from-file=./secret/basic_auth.ini mlflow-db-config
run-apply:
	kubectl apply -k . -n $(NAMESPACE)