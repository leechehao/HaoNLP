from typing import Optional

import mlflow
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue


def check_missing_value(cfg: DictConfig) -> None:
    missings = OmegaConf.missing_keys(cfg)
    if missings:
        raise MissingMandatoryValue(f"Missing mandatory value: {missings}")


def mlflow_setup(
    tracking_uri: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.start_run(run_id=run_id)


def log_config(cfg: DictConfig, hycfg: DictConfig, artifact_file: str) -> None:
    # mlflow.log_param("module_class_path", cfg.task._target_)
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(cfg_dict, artifact_file)

    if "experiment" in hycfg.runtime.choices:
        for d in hycfg.runtime.config_sources:
            if d.provider == "main":
                expcfg_path = f"{d.path}/experiment/{hycfg.runtime.choices.experiment}.yaml"
                mlflow.log_artifact(expcfg_path)
