# lume-services-deployment

A repository for holding the manifest files for deploying the lume services infrastructure to a Kubernetes cluster.

The deployment uses non-cloud prefect as the workflow orchestration software for running models and simulations. PostgreSQL is used as the backend storage layer for prefect.

The MySQL and MongoDB directories were for a model DB and a results DB respectively. However, with the use of MLflow the
MySQL DB is no longer required and can most likely be removed. MongoDB was there only for an initial prototype and can
be modified or removed as well as needed.
