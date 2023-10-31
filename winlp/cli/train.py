import argparse

import yaml
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
    trainer = instantiate(cfg.trainer, logger=mlf_logger)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
