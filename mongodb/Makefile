NAMESPACE ?= resultsdb

operator:
	curl https://raw.githubusercontent.com/percona/percona-server-mongodb-operator/main/deploy/rbac.yaml > mongodb-operator-rbac.yaml
	curl https://raw.githubusercontent.com/percona/percona-server-mongodb-operator/main/deploy/crd.yaml > mongodb-operator-crd.yaml
	curl https://raw.githubusercontent.com/percona/percona-server-mongodb-operator/main/deploy/operator.yaml > mongodb-operator.yaml

run-apply:
	kubectl apply -k .
