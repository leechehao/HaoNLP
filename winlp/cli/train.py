import hydra
import mlflow
import torch
import lightning as L
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from winlp.core import utils


torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ===== 檢查參數缺失值 =====
    utils.check_missing_value(cfg)

    # ===== 設定種子 =====
    L.seed_everything(cfg.seed, workers=True)

    # ===== 建立資料模組 =====
    data_module = instantiate(cfg.dataset)
    data_module.setup("fit")

    # ===== 建立模型 =====
    model = instantiate(cfg.task, label_list=data_module.label_list)

    # ===== 設定 MLflow logger =====
    mlf_logger = instantiate(cfg.trainer.logger)
    utils.mlflow_setup(cfg.trainer.logger.tracking_uri, mlf_logger.experiment_id, mlf_logger.run_id)
    utils.log_config(cfg, HydraConfig.get(), artifact_file="config.yaml")

    # ===== 設定 Trainer =====
    trainer = instantiate(cfg.trainer, logger=mlf_logger)

    # ===== 訓練 =====
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    # ===== 上傳 onnx 模型 =====
    logged_model = mlflow.pytorch.load_model(f"runs:/{mlflow.active_run().info.run_id}/model")
    logged_model.log_onnx_model()

    # ===== 測試集評估 =====
    if cfg.test:
        trainer.test(logged_model, data_module)


if __name__ == "__main__":
    # python winlp/cli/train.py +experiment=token_classification/chest_ct_1
    main()
