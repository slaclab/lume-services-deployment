NAMESPACE ?= prefect

operator:
	curl https://raw.githubusercontent.com/zalando/postgres-operator/master/manifests/configmap.yaml > configmap.yaml
	curl https://raw.githubusercontent.com/zalando/postgres-operator/master/manifests/operator-service-account-rbac.yaml > postgres-operator-services-account-rbac.yaml
	curl https://raw.githubusercontent.com/zalando/postgres-operator/master/manifests/postgres-operator.yaml > postgres-operator.yaml
	curl https://raw.githubusercontent.com/zalando/postgres-operator/master/manifests/api-service.yaml > postgres-api-service.yaml

run-apply:
	kubectl apply -k .
