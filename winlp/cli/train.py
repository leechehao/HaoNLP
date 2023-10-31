from typing import Optional

import mlflow
import hydra
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 待補檢查必須參數
    # print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.dataset)

    model = instantiate(cfg.task, num_labels=data_module.num_labels)

    mlf_logger = instantiate(cfg.trainer.logger)
    mlflow_setup(cfg.trainer.logger.tracking_uri, mlf_logger.experiment_id, mlf_logger.run_id)

    log_config(cfg, artifact_file="config.yaml")

    trainer = instantiate(cfg.trainer, logger=mlf_logger)

    trainer.fit(model, data_module)


def mlflow_setup(
    tracking_uri: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.start_run(run_id=run_id)


def log_config(cfg: DictConfig, artifact_file: str) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(cfg_dict, artifact_file)


if __name__ == "__main__":
    main()
