NAMESPACE ?= modeldb

operator-yaml:
	curl https://raw.githubusercontent.com/mysql/mysql-operator/trunk/deploy/deploy-operator.yaml > mysql-operator.yaml
	curl https://raw.githubusercontent.com/mysql/mysql-operator/trunk/deploy/deploy-crds.yaml > mysql-crd.yaml

run-apply:
	kubectl apply -k .

