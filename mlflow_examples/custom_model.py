import os  # dont do this in production

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

os.environ["AWS_DEFAULT_REGION"] = "eu-west-3"
os.environ["AWS_REGION"] = "eu-west-3"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:30008"
# tracking uri
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:30007"

from mlflow import MlflowClient
from mlflow.server import get_app_client
import mlflow
from mlflow.models import infer_signature

import matplotlib.pyplot as plt

import numpy as np


class MyModelType(mlflow.pyfunc.PythonModel):  # this can wrap around a lume epics model
    def __init__(self, model_name):
        self.model_name = model_name

    # this function is called when the model is loaded using pyfunc.load_model
    def predict(self, context, input, **kwargs):
        return self.predict_internal(input, **kwargs)

    # this function is the true predict function, but we need the context parameter to be able to use the model with mlflow
    def predict_internal(self, input, **kwargs):
        return np.array(input) ** 2

    def inverse_predict_internal(self, input, **kwargs):
        return np.sqrt(input)

    def save_model(self):
        with open(f"{self.model_name}.txt", "w") as f:
            f.write("model saved")

    def load_model(self):
        with open(f"{self.model_name}.txt", "r") as f:
            return f.read()


with mlflow.start_run() as run:  # you can use run_name="test1" to give a name to the run otherwise it will a random name
    input_sample = [1, 2, 3, 4, 5]
    input_sample = np.array(input_sample)

    model = MyModelType("model1")
    # model.save_model() # no need to save the model since it is saved in log_model
    mlflow.log_param("model_name", model.model_name)
    mlflow.log_param("dummy_param1", "dummy_value1")
    mlflow.log_param("dummy_param2", 0.33)
    for i in range(10):
        mlflow.log_metric("metric1", (i / 10) ** 2)
        mlflow.log_metric("metric2", (i / 10) ** 3)
        mlflow.log_metric("loss", (1 / (i + 0.1) + np.random.normal(0, 0.1)))

    # lets make some pretty graphs to stor

    graph = plt.figure()
    plt.plot(range(100), [(i / 10) ** 2 for i in range(100)])
    mlflow.log_figure(graph, "figures/metric1.png")

    # alternative way to log a figure
    graph = plt.figure()
    plt.plot(range(100), [(i / 10) ** 3 for i in range(100)])
    graph.savefig("metric2.png")
    mlflow.log_artifact("metric2.png", artifact_path="figures")

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model_files",
        python_model=model,
        signature=infer_signature(input_sample, model.predict_internal(input_sample)),
        input_example=input_sample,
        # registered_model_name="model1",  # this will automatically register the model and iterate the version
    )

    # if you wanna log the model without the wrapper
    model.save_model()
    mlflow.log_artifact(
        f"{model.model_name}.txt", artifact_path="model_files_no_mlflow"
    )

    # set some tags
    mlflow.set_tag("tag1", "tag_value1")
    mlflow.set_tag("tag2", "tag_value2")
    mlflow.set_tag("tag3", "tag_value3")

model_uri = model_info.model_uri
# model_uri = 'runs:/6bfc6f63045c495c91206f2728acfa5a/model1'

input_sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
input_sample = np.array(input_sample)

loaded_model = mlflow.pyfunc.load_model(model_uri)
print(type(loaded_model))
print("making prediction using the loaded model")
print(loaded_model.predict(input_sample))

unwrapped_model = loaded_model.unwrap_python_model()
print(type(unwrapped_model))
print("making prediction using the unwrapped model .predict_internal()")
print(unwrapped_model.predict_internal(input_sample))
# or
print("making prediction using the unwrapped model .predict()")
print(
    unwrapped_model.predict(None, input_sample)
)  # the None parameter is the context parameter, it is not used in this example


# get @latest version of the model
client = MlflowClient()
# get by alias
model_version = client.get_model_version_by_alias("model1", "deployment")
print(model_version)

# say we are ready for production we want to register the model
# client = MlflowClient()
# # client.create_registered_model("model1") # if the model is not registered yet
# client.create_model_version(
#     name="model1", source=model_uri, run_id=run.info.run_id, aliases=["model1-1.0.0"]
# )
# you can also do this using the mlflow ui


# # inspect model tags
# client = MlflowClient()
# tags = client.get_run(run.info.run_id).data.tags
# print(tags)
# # these are stored in the mlflow database (postgres in this case) and can be used to filter, mark ready for deployment, etc
# # for examples lets search for all models that have the tag "tag1" with value "tag_value1"

# all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
# runs = client.search_runs(experiment_ids=all_experiments, filter_string="tags.tag1 = 'tag_value1'")

# print("runs with tag1 = tag_value1:")
# print(runs)
