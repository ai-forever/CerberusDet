import hashlib
import json
import os
import shutil
from typing import List

import mlflow
import torch
from loguru import logger
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


def init_mlflow(mlflow_url):
    assert mlflow_url is not None, "mlflow_url must be set"
    if mlflow_url in ["localhost", "local"]:
        return
    envs = os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), os.getenv("MLFLOW_S3_ENDPOINT_URL")
    assert all(env is not None for env in envs), "Credentials for s3 are required"

    mlflow.set_tracking_uri(mlflow_url)

    return mlflow_url


class MLFlowLogger:
    """
    Log training runs, datasets, models, and predictions to MLFlow
    """

    def __init__(self, opt, hyp=None):
        # Pre-training routine --
        self.artifact_path = "states"
        self.hyp = hyp

        init_mlflow(opt.mlflow_url)

        experiment_name = f"{opt.experiment_name}"
        run_name = f"{opt.name}"

        runs = mlflow.search_runs(experiment_names=[experiment_name])

        if len(runs) == 0:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            client = MlflowClient()
            experiment_id = runs["experiment_id"][0]
            run_ids = runs["run_id"]
            run_names = [
                client.get_run(run_id).data.tags["mlflow.runName"]
                for run_id in run_ids
                if "mlflow.runName" in client.get_run(run_id).data.tags
            ]
            if run_name in run_names:
                # if such run name already exists
                n_prev_runs = len([1 for name in run_names if run_name in name])
                run_name = f"{run_name}_{n_prev_runs}"

            mlflow.set_experiment(experiment_name=experiment_name)

        mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

        self.mlflow_experiment = experiment_id
        self.mlflow_run = mlflow_run.info.run_id

        logger.info(
            f"MLFlow: view at {opt.mlflow_url}."
            f"\n Experiment name, id: {experiment_name}, {experiment_id}."
            f"\n Run name, id: {run_name}, {mlflow_run.info.run_id}."
        )

        for params in [self.hyp, vars(opt)] if self.hyp is not None else [vars(opt)]:
            for k, v in params.items():
                value = json.dumps(v) if isinstance(v, list) or isinstance(v, dict) else v
                mlflow.log_param(key=k, value=value)

    @staticmethod
    def finish_run():
        mlflow.end_run()

    def save_artifacts(self, local_dir_path):
        mlflow.log_artifacts(local_dir_path, artifact_path=self.artifact_path)

    def save_artifact(self, f_name):
        mlflow.log_artifact(f_name, artifact_path=self.artifact_path)

    def log_model(self, model_path):
        mlflow.log_artifact(model_path, artifact_path=self.artifact_path)

    @staticmethod
    def log_model_signature(model, im_size, device, relative_uri="model"):
        imgsz = (1, 3, im_size, im_size)
        sample_input = torch.zeros(*imgsz).to(device).half()
        sample_output_all_tasks = model(sample_input)

        dict_of_arrays = {}
        for task_name, task_output in sample_output_all_tasks.items():
            if isinstance(task_output, tuple) and len(task_output) == 2:
                dict_of_arrays[f"output_{task_name}"] = task_output[0].detach().cpu().numpy()
            else:
                for n_out, out in enumerate(task_output):
                    dict_of_arrays[f"output_{task_name}_{n_out}"] = out.detach().cpu().numpy()

        signature = infer_signature({"images": sample_input.cpu().numpy()}, dict_of_arrays)

        # Upload model and it's checksum
        mlflow.pytorch.log_model(model, relative_uri, signature=signature)

    @staticmethod
    def log_best_model_md5(best_model_path, relative_uri="model/data"):
        success = upload_mlflow_checksum(best_model_path, relative_uri)
        if not success:
            logger.error("Can not upload model checksum to mlflow")

    @staticmethod
    def log_params(params):
        # params: A dict containing {param: value} pairs
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics, step):
        mlflow.log_metrics(metrics, step=step)


def upload_mlflow_checksum(model_path, relative_uri) -> bool:
    """Loads model, calculates MD5 sum and uploads as [name].md5 file.

    Args:
        model_path (str): url to MLFlow model artefacts
        relative_uri (str, optional): Path to upload checksum. Defaults to 'model/data'.

    Returns:
        _type_: True if success, False otherwise
    """

    if not os.path.exists(model_path):
        logger.error(f"File {model_path} does not exist")
        return False

    md5sum = hashlib.md5(open(model_path, "rb").read()).hexdigest()
    md5file = model_path.replace(".pt", ".md5")
    with open(md5file, "w") as f:
        f.write(md5sum)

    logger.info("uploading md5file artifact")
    mlflow.log_artifact(md5file, relative_uri)

    return True


def list_artifacts(artifacts: list, client: MlflowClient, run_id: str, path=None):

    run_artifacts = client.list_artifacts(run_id, path)
    for artifact in run_artifacts:
        if not artifact.is_dir:
            artifacts.append(artifact)
        else:
            list_artifacts(artifacts, client, run_id, artifact.path)


def attempt_mlflow_download(model: str) -> str:
    """Trying to download MLFlow model if possible

    Args:
        model (str): Model id or path. For example:
        "models:/ClothesDetector/1" or "models:/ClothesDetector/1/states/best.pt"

    Returns:
        str: Path to local file
    """
    model_path = model.replace("models:/", "")

    model_name = model_path.split("/")[0]
    model_version = model_path.split("/")[1]
    model_file_name = None

    if ".pt" == model_path[-3:] or ".pth" == model_path[-4:]:
        model_file_name = "/".join(model_path.split("/")[2:])

    client = mlflow.MlflowClient()
    finish_model_run_id = None

    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.version == model_version:
            finish_model_run_id = mv.run_id

    if finish_model_run_id is None:
        raise ValueError(f"Can't find model {model_name} with version {model_version}")

    run_artifacts: List[mlflow.entities.FileInfo] = []
    list_artifacts(run_artifacts, client, finish_model_run_id)

    ckpt_models = []
    for artifact in run_artifacts:
        if not artifact.is_dir and (".pt" == artifact.path[-3:] or ".pth" == artifact.path[-4:]):
            ckpt_models.append(artifact.path)

    assert len(ckpt_models), f"Can't find any checkoints for model {model_name}/{model_version}"

    artifact_to_download = None
    if model_file_name:
        for ckpt_path in ckpt_models:
            if model_file_name == ckpt_path or model_file_name in ckpt_path:
                artifact_to_download = ckpt_path
        assert artifact_to_download is not None, f"Can't find ckpt with name {model_file_name}"
    elif len(ckpt_models) == 1:
        artifact_to_download = ckpt_models[0]
    else:
        logger.warning(
            f"For model {model_name}/{model_version} found {len(ckpt_models)} possible checkpoints. "
            f"'best.pt' will be downloaded"
        )
        for ckpt_path in ckpt_models:
            if "best.pt" in ckpt_path:
                artifact_to_download = ckpt_path
                break
        assert artifact_to_download is not None, "Can't find ckpt with best.pt name. Specify ckpt name."

    new_filename = os.path.join(os.getcwd(), f"{model_name}_v{model_version}.pt")
    local_path = client.download_artifacts(finish_model_run_id, artifact_to_download, os.getcwd())

    shutil.move(local_path, new_filename)
    logger.info(f"Ckpt {artifact_to_download} downloaded to {new_filename}")

    return new_filename
