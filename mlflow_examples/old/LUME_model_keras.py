import os  

# dont do this in production
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["AWS_DEFAULT_REGION"] = "eu-west-3"
os.environ["AWS_REGION"] = "eu-west-3"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://172.24.5.222:9020"
# tracking uri
os.environ["MLFLOW_TRACKING_URI"] = "https://ard-mlflow.slac.stanford.edu"

from mlflow import MlflowClient
from mlflow.server import get_app_client
import mlflow
from mlflow.models import infer_signature

import matplotlib.pyplot as plt

import keras

import numpy as np
import pandas as pd

#### Lume Model Keras example (also stolen from docs)
from lume_model.models import KerasModel
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable


class ExampleModel(): # no need to inherit from LUMEBaseModel but we are wrapping around a lume model
    def __init__(self):   
        # exemplary model definition
        
        inputs = [keras.Input(name="input1", shape=(1,)), keras.Input(name="input2", shape=(1,))]
        outputs = keras.layers.Dense(1, activation=keras.activations.relu)(keras.layers.concatenate(inputs))
        base_model = keras.Model(inputs=inputs, outputs=outputs)
        # variable specification
        input_variables = [
            ScalarInputVariable(name=inputs[0].name, default=0.1, value_range=[0.0, 1.0]),
            ScalarInputVariable(name=inputs[1].name, default=0.2, value_range=[0.0, 1.0]),
        ]
        output_variables = [
            ScalarOutputVariable(name="output"),
        ]

        # creation of KerasModel
        self.example_model = KerasModel(
            model=base_model,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        
    def evaluate(self, input_dict):
        return self.example_model.evaluate(input_dict)
    
    def serialize(self):
        return self.example_model.d


class MyModelWrapper(mlflow.pyfunc.PythonModel):  # this can wrap around a lume epics model
    def __init__(self, model_name):
        self.model_name = model_name
        # definition of the keras model
        self.lume_model = ExampleModel()

    # this function is called when the model is loaded using pyfunc.load_model
    def predict(self, context, input, **kwargs):
        return self.predict_internal(input, **kwargs)

    # this function is the true predict function, but we need the context parameter to be able to use the model with mlflow
    def predict_internal(self, input, **kwargs):
        print(type(input))
        print(input.shape)
        print(type(self.lume_model.evaluate(input)))
        print(self.lume_model.evaluate(input))
        return self.lume_model.evaluate(input)

    def inverse_predict_internal(self, input, **kwargs):
        return np.sqrt(input)

    def save_model(self):
        with open(f"{self.model_name}.txt", "w") as f:
            f.write("model saved")

    def load_model(self):
        with open(f"{self.model_name}.txt", "r") as f:
            return f.read()
        
    def unwrap_python_model(self):
        return self.lume_model
    
    def make_serializable(self, model):
        return model
    

with mlflow.start_run() as run:  # you can use run_name="test1" to give a name to the run otherwise it will a random name
    

    model = MyModelWrapper("model1")
    input_sample = model.lume_model.example_model.random_input(n_samples=10)
    # model.save_model() # no need to save the model since it is saved in log_model
    mlflow.log_param("model_name", model.model_name)
    mlflow.log_param("dummy_param1", "dummy_value1")
    mlflow.log_param("dummy_param2", 0.33)
    for i in range(10):        
        mlflow.log_metric("metric1", (i / 10) ** 2 , step=i)
        mlflow.log_metric("metric2", (i / 10) ** 3 , step=i)
        mlflow.log_metric("loss", (1 / (i + 0.1) + np.random.normal(0, 0.1)) , step=i)

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
        signature=infer_signature(input_sample, model.lume_model.evaluate(input_sample)),
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

input_sample = model.lume_model.example_model.random_input(n_samples=10)

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

# set alias


# get @latest version of the model
client = MlflowClient()
# register model and set alias

try:
    client.create_registered_model("model1")
except:
    pass

result = client.create_model_version(
    name="model1",
    source=model_uri,
    run_id=run.info.run_id,
)
client.set_registered_model_alias("model1", "champion", result.version)

# get by alias
model_version = client.get_model_version_by_alias("model1", "champion")
print(model_version)