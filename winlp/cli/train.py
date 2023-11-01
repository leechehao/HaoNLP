from typing import Optional

import mlflow
import hydra
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from omegaconf.errors import MissingMandatoryValue
from hydra.utils import instantiate


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    check_missing_value(OmegaConf.to_container(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.dataset)

    model = instantiate(cfg.task, num_labels=data_module.num_labels)

    mlf_logger = instantiate(cfg.trainer.logger)
    mlflow_setup(cfg.trainer.logger.tracking_uri, mlf_logger.experiment_id, mlf_logger.run_id)

    log_config(cfg, HydraConfig.get(), artifact_file="config.yaml")

    trainer = instantiate(cfg.trainer, logger=mlf_logger)

    trainer.fit(model, data_module)


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
    # OmegaConf.update(cfg, "hydra", hycfg, force_add=True)
    # cfg_dict = OmegaConf.to_container(cfg)
    # mlflow.log_dict(cfg_dict, artifact_file)

    # hycfg_dict = OmegaConf.to_container(hycfg)
    # mlflow.log_dict(hycfg_dict, artifact_file)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(cfg_dict, artifact_file)

    for d in hycfg.runtime.config_sources:
        if d.provider == "main":
            expcfg_path = f"{d.path}/experiment/{hycfg.runtime.choices.experiment}.yaml"
            mlflow.log_artifact(expcfg_path)

 
if __name__ == "__main__":
    main()
