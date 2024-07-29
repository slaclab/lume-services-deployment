# Prefect

The prefect server defines a local non-cloud based deployment of [prefect](https://www.prefect.io/) with a PostgreSQL 
data storage layer. 

The frontend for the server is available at https://ard-modeling-service.slac.stanford.edu with access limited to SLAC
and NERSC IP addresses as defined in the endpoints file.

The worker is created to listen for tasks given to the `kubernetes-work-pool`. It has a few extra permissions so that it
can create and monitor pods from MLflow based workflows.
