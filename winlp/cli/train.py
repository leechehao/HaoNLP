import hydra
import mlflow
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from winlp.core import utils


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    utils.check_missing_value(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.dataset)

    model = instantiate(cfg.task, num_labels=data_module.num_labels)

    mlf_logger = instantiate(cfg.trainer.logger)
    utils.mlflow_setup(cfg.trainer.logger.tracking_uri, mlf_logger.experiment_id, mlf_logger.run_id)
    utils.log_config(cfg, HydraConfig.get(), artifact_file="config.yaml")
    
    trainer = instantiate(cfg.trainer, logger=mlf_logger)

    trainer.fit(model, data_module)

    if cfg.test:
        logged_model = mlflow.pytorch.load_model(f"runs:/{mlflow.active_run().info.run_id}/model")
        trainer.test(logged_model, data_module)

if __name__ == "__main__":
    # python winlp/cli/train.py +experiment=token_classification/chest_ct_1
    main()
