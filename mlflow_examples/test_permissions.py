import os  # dont do this in production

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

from mlflow import MlflowClient
from mlflow.server import get_app_client

tracking_uri = "http://localhost:30007"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
# auth_client.create_user(username="user2", password="pw2")

client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.get_experiment_by_name("test2")
experiment_id = experiment.experiment_id
auth_client.update_experiment_permission(  # use create_experiment_permission to create a new permission
    experiment_id=experiment_id, username="user2", permission="NO_PERMISSIONS"
)  # this iser will not be able to see the experiment
auth_client.update_experiment_permission(
    experiment_id=experiment_id, username="user1", permission="READ"
)  # this user will be able to see the experiment but not edit or delete it

experiment = client.get_experiment_by_name("test2")
experiment_id = experiment.experiment_id
ep = auth_client.get_experiment_permission(experiment_id, "user1")
print(f"experiment_id: {ep.experiment_id}")
print(f"user_id: {ep.user_id}")
print(f"permission: {ep.permission}")


# permisson levels lookup
# "NO_PERMISSIONS" - no permissions, cant see the experiment
# "READ" - can see the experiment
# "EDIT" - can see and manage the experiment - cannot delete the experiment
# "MANAGE" - can see, manage and change permissions of the experiment
