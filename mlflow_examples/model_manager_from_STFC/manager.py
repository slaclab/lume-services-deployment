# model manager uses the outdates lifecycle  api (superseeded by aliases)
# The below can be handled by a periodic prefect task that checks for new models and deploys them or a 
# similar solution to the one below.

import os
import json
import shutil
import re
import tempfile
import time

import yaml
import mlflow
from mlflow.tracking import MlflowClient
from model_manager.logging_conf import make_logger

import docker


class ModelManager:
    """
    Model deployment manager for for mlflow models
    """

    def __init__(self) -> None:
        self.logger = make_logger()
        if os.name == "nt":
            self.logger.warning("On Windows - using dev settings!")
            secret_file = json.load(
                open("./model_manager/top-secret", encoding="UTF-8")
            )
        else:
            # get file from docker secrets
            self.logger.warning("On Linux - using prod settings!")
            secret_file = json.load(open("/run/secrets/s3-creds", encoding="UTF-8"))

        # setup mlflow stuff
        os.environ["AWS_DEFAULT_REGION"] = secret_file["AWS_DEFAULT_REGION"]
        os.environ["AWS_REGION"] = secret_file["AWS_DEFAULT_REGION"]
        os.environ["AWS_ACCESS_KEY_ID"] = secret_file["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_file["AWS_SECRET_ACCESS_KEY"]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = secret_file["MLFLOW_S3_ENDPOINT_URL"]

        mlflow.set_tracking_uri(secret_file["MLFLOW_SERVER"])

        # check if models folder exists
        if not os.path.exists("./models"):
            self.logger.info("Creating models folder")
            os.mkdir("./models")

        # check if ./models/models.config exists
        if not os.path.exists("./models/models.config"):
            self.logger.info("Creating models.config file")
            with open("./models/models.config", "w") as f:
                init_config = "model_config_list {\n}"
                f.write(init_config)
        # if the file exists, check if it is empty
        if os.stat("./models/models.config").st_size == 0:
            self.logger.info(
                "models.config is empty, writing init config, minimal config"
            )
            with open("./models/models.config", "w") as f:
                init_config = "model_config_list {\n}"
                f.write(init_config)

        # lint the models.config file to make sure it is valid

        if not self.lint_models_config():
            raise Exception("models.config is not valid!")

    def lint_models_config(self) -> bool:
        """Lint the models.config file to make sure it is valid"""
        with open("./models/models.config", "r") as f:
            config = f.read()
        # check if the config is valid
        if re.search(r"model_config_list {", config) is None:
            self.logger.error("models.config is not valid!")
            return False

        # count the number of curly brackets and make sure they are equal

        elif config.count("{") != config.count("}"):
            self.logger.error("models.config is not valid, unbalanced curly brackets!")
            return False

        else:
            self.logger.info("models.config is valid!")
            return True

    def get_models(self) -> list:
        """Get a list of all models in production from mlflow server"""
        client = MlflowClient()
        model_cnt = 0
        versions_cnt = 0
        prod_cnt = 0
        prod_models = []

        for registered_models in client.search_registered_models():
            model_cnt += 1
            for reg_model in dict(registered_models)["latest_versions"]:
                versions_cnt += 1
                if dict(reg_model)["current_stage"] == "Production":
                    prod_cnt += 1
                    prod_models.append(dict(reg_model))
        self.logger.info(
            "Found %s versions of %s models of which %s are in Production",
            versions_cnt,
            model_cnt,
            prod_cnt,
        )

        return prod_models

    def download_model(self, model: dict, download_path: str = "./tmp") -> str:
        """
        Download model from mlflow server and deploy it to model server
        and return the path to the model
        """
        model_name = model["name"]
        model_version = model["version"]
        model_path = f"{download_path}/{model_name}/{model_version}"
        self.logger.debug(
            "Downloading model %s version %s to %s",
            model_name,
            model_version,
            model_path,
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{model_version}",
            dst_path=model_path,
        )

        return model_path

    def download_models(
        self, models: list = None, download_path: str = "./tmp"
    ) -> list:
        """Download all models in production from mlflow server and return a list of paths to the models"""

        if models is None:
            models = self.get_models()

        model_paths = []
        for model in models:
            model_paths.append(self.download_model(model, download_path))
        return model_paths

    def deploy(
        self,
        model_paths: list,
    ) -> None:
        """Process all models in production"""
        changed = False
        for model_path in model_paths:
            # in model path load yaml file with MLmodel
            model_meta = yaml.safe_load(open(f"{model_path}/MLmodel", encoding="UTF-8"))
            # if tensorflow key in "flavour" then deploy with tensorflow
            if "tensorflow" in model_meta["flavors"].keys():
                self.logger.info("Deploying tensorflow model %s", model_path)
                # need some mdoel format check here
                self.__append_config(
                    model_meta, "./models/models.config", model_path, "tensorflow"
                )

            elif "keras" in model_meta["flavors"].keys():
                self.logger.info("Deploying keras model %s", model_path)
                self.__append_config(
                    model_meta, "./models/models.config", model_path, "keras"
                )
                # self.__move_to_models(model_path)

            elif "pytorch" in model_meta["flavors"].keys():
                self.logger.info("Deploying pytorch model %s", model_path)

                # if linux
                if os.name != "nt":
                    client = docker.DockerClient(base_url="unix://var/run/docker.sock")
                elif os.name == "nt":
                    client = docker.from_env()

                # check if torchserve is running, the container should have a name with torch_model_server
                server_running = False
                if len(
                    client.containers.list(
                        filters={"name": "torchserve_torch_model_server"}
                    )
                ):
                    server_running = True
                    server_info = client.containers.list(
                        filters={"name": "torchserve_torch_model_server"}
                    )[0]
                    server_id = server_info.id
                    self.logger.debug("Torchserve is running")

                # copy mar file to /mnt/powervault/athena_share/torchserve_models/ from model_path/mar_model
                # check if mar file exists in model_path
                if os.path.exists(f"{model_path}/mar_model"):
                    # check for .mar file and get the name
                    mar_file = [
                        f
                        for f in os.listdir(f"{model_path}/mar_model")
                        if f.endswith(".mar")
                    ]
                    if len(mar_file) == 1:
                        mar_file = mar_file[0]
                    # copy mar file to /mnt/powervault/athena_share/torchserve_models/
                    if os.name != "nt":
                        dest = "/models-torchserve/"
                    elif os.name == "nt":
                        dest = "./tmp/torchserve_models/"
                        # check dir exists
                        if not os.path.exists(dest):
                            os.mkdir(dest)
                    try:
                        shutil.move(
                            f"{model_path}/mar_model/{mar_file}",
                            dest,
                        )
                        # restart torchserve
                        changed = True
                    # file already exists
                    except shutil.Error as error:
                        self.logger.warning(
                            "Model already exists in models dir %s", error
                        )
                        continue
                else:
                    self.logger.error(
                        "No mar file found in %s, skipping pytorch model deployment",
                        model_path,
                    )
                    continue
                # take this out to a config file!

                # check if model_name.mar file exists in /mnt/powervault/athena_share/torchserve_models/
                # if not, copy it form
            else:
                self.logger.error(
                    "Model %s is not supported, skipping model deployment", model_path
                )
                continue

        if changed and server_running:
            self.logger.info("Restarting torchserve")
            client.containers.get(server_id).restart()
            # wait for torchserve to restart
            while True:
                try:
                    client.containers.get(server_id)
                    break
                except docker.errors.NotFound:
                    self.logger.debug("Torchserve not yet running")
                    time.sleep(2)

        self.__clean_up()

    def __append_config(
        self,
        model_meta: dict,
        model_conf: str,
        model_path: str,
        model_platform: str,
        models_path: str = "./models",
    ) -> None:
        """Append model config to config file
        Config should look like this:
        model_config_list: {
            config: {
                name: "model_name",
                base_path: "model_path",
                model_platform: "model_platform",
        }
        """
        # get model name form model_dir, split on / and get last two elements (model/version)
        model_name = model_path.split("/")[-2]
        model_version = model_path.split("/")[-1]
        # str to append to config file
        model_config = ""
        model_config += "\tconfig: {\n\t\t"
        # model name
        model_config += f'name: "{model_name}",\n\t\t'

        if "saved_model_dir" in model_meta["flavors"][model_platform].keys():
            # tensorflow
            self.logger.debug("Moving tensorflow model to models dir")
            model_data = "saved_model_dir"

        elif "data" in model_meta["flavors"][model_platform].keys():
            self.logger.debug("Moving keras model to models dir")
            model_data = "data"

        new_model_path = f"{models_path}/{model_name}/{model_version}"
        if model_platform == "tensorflow":
            src_dir = f"{model_path}/{model_meta['flavors'][model_platform][model_data]}/model"
        elif model_platform == "keras":
            src_dir = f"{model_path}/{model_meta['flavors'][model_platform][model_data]}/model"
        try:
            shutil.move(
                src_dir,
                new_model_path,
            )
        # file already exists
        except shutil.Error as error:
            self.logger.warning("Model already exists in models dir %s", error)
            return None

        # model path should be model_path/version
        # drop . from new_model_path
        new_model_path_no_ver = f"{models_path}/{model_name}"
        model_config += f'base_path: "{new_model_path_no_ver.replace(".","")}",\n\t\t'
        # model platform
        model_config += 'model_platform: "tensorflow",\n\t'
        # closing bracket
        model_config += "\n\t}\n"

        # load models.config
        with open(model_conf, "r", encoding="utf-8") as f:
            config = f.read()

        # check if model is already in config
        if re.search(model_name, config):
            self.logger.warning("Model %s already in config", model_name)
        # find all occurences of }
        else:
            brackets = [m.start() for m in re.finditer("}", config)]
            # insert model_config before last }
            config = config[: brackets[-1]] + model_config + config[brackets[-1] :]

            # save
            with open(model_conf, "w", encoding="utf-8") as f:
                f.write(config)

        return None

    def __clean_up(
        self,
        models_path: str = "./models",
        model_config: str = "./models/models.config",
    ) -> None:
        """Delete all models from models from model_path that are not in model_config"""
        # load models.config
        with open(model_config, "r", encoding="utf-8") as f:
            config = f.read()

        # get all models paths by splitting by new line and getting all lines that contain base_path
        config = config.splitlines()
        config = [line for line in config if "base_path" in line]
        config = [
            "." + line.split(":")[1].strip().replace('"', "").replace(",", "")
            for line in config
        ]

        # now get list of all models in models_path (depth 1) -> ./models/x
        models = [f"./models/{x}" for x in os.listdir(models_path)]
        # ignore models.config
        models = [x for x in models if x != model_config]

        for line in models:
            if line not in config:
                self.logger.warning("Cleanup - Deleting %s", line)
                shutil.rmtree(line)
        # check if paths exist
