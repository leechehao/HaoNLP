import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from winlp.core import utils


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    utils.check_missing_value(cfg)

    data_module = instantiate(cfg.dataset)

    trainer = instantiate(cfg.trainer, logger=False)
    
    mlflow.set_tracking_uri(cfg.trainer.logger.tracking_uri)
    logged_model = mlflow.pytorch.load_model(f"runs:/{cfg.run_id}/model")
    trainer.test(logged_model, data_module)

if __name__ == "__main__":
    # python winlp/cli/train.py +experiment=token_classification/chest_ct_1
    main()
