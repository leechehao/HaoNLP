import os

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from winlp.core import utils


torch.set_float32_matmul_precision("high")

winlp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_config_path = os.path.join(winlp_dir, "../winlp_conf")

CONFIG_PATH = os.getenv("WINLP_CONFIG_PATH", default_config_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    utils.check_missing_value(cfg)

    data_module = instantiate(cfg.dataset)

    trainer = instantiate(cfg.trainer, logger=False)

    mlflow.set_tracking_uri(cfg.trainer.logger.tracking_uri)
    # module_name, class_name = cfg.task._target_.rsplit(".", 1)
    # checkpoint_path = mlflow.artifacts.download_artifacts(f"runs:/{cfg.run_id}/model/state_dict.pth")
    # logged_model = getattr(importlib.import_module(module_name), class_name).load_from_checkpoint(checkpoint_path)
    logged_model = mlflow.pytorch.load_model(f"runs:/{cfg.run_id}/model")
    trainer.test(logged_model, data_module)


if __name__ == "__main__":
    # python winlp/cli/test.py +experiment=token_classification/chest_ct_1
    main()
