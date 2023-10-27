from typing import Dict, Sequence, Tuple, Type, Union

from weakref import proxy

import lightning.pytorch as pl
import transformers
import mlflow
from torch.optim import Optimizer, AdamW
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from lightning.pytorch.loggers import MLFlowLogger


class Module(pl.LightningModule):
    """
    基於 PyTorch Lightning 的 Module，用於訓練和測試的深度學習模型模板。

    主要處理包括模型初始化、度量指標配置、優化器和學習速率調度器的配置，以及訓練期間的回調函數配置。
    """

    def __init__(
        self,
        downstream_model_type: Type[_BaseAutoModelClass],
        pretrained_model_name_or_path: str,
        num_labels: int,
        monitor: str,
        mode: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.0,
    ) -> None:
        """
        初始化 Module。

        Args:
            downstream_model_type (Type[_BaseAutoModelClass]): 下游任務模型的類型。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            num_labels (int): 分類任務的標籤數量。
            monitor (str): 要監控的指標名稱。
            mode (str): 監控指標的模式，例如 `min`、`max`。
            learning_rate (float, optional): 學習率。預設為 1e-3。
            weight_decay (float, optional): 權重衰減，用於正則化和防止模型過擬合。預設為 1e-2。
            warmup_ratio (float, optional): 預熱比例，在訓練初期學習率逐漸提升階段所佔的總訓練步驟比例。預設為 0.0。
        """
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model = downstream_model_type.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
        )
        self.num_labels = num_labels
        self.monitor = monitor
        self.mode = mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

    def setup(self, stage: str) -> None:
        """
        在每個階段開始時調用，用於配置模型相關的設置。

        Args:
            stage (str): 當前的訓練階段。
        """
        # self.logger.log_hyperparams()  # 待補 hydra model_config
        self.configure_metrics()

    def configure_metrics(self) -> None:
        """
        覆寫以配置用於訓練和驗證的度量指標。
        """
        pass

    def configure_optimizers(self) -> Tuple[Sequence[Optimizer], Sequence[Dict]]:
        """
        配置用於訓練模型的優化器和學習率調度器。

        Returns:
            Tuple[Sequence[Optimizer], Sequence[Dict]]: 返回一個包含優化器和學習率調度器設置的元組。
        """
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.trainer.estimated_stepping_batches * self.warmup_ratio,
        )
        scheduler = {"scheduler": self.scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [scheduler]

    def configure_callbacks(self) -> Union[Sequence[pl.callbacks.Callback], pl.callbacks.Callback]:
        """
        配置訓練過程中使用的回調函數。

        Returns:
            Union[Sequence[pl.callbacks.Callback], pl.callbacks.Callback]: 一個或多個回調函數。
        """
        calback = MLflowModelCheckpoint(logger=self.logger, monitor=self.monitor, mode=self.mode)
        return [calback]


class MLflowModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    專為 MLflow 整合的 ModelCheckpoint。

    主要功能是在訓練過程中選擇最佳模型檢查點，然後將該模型保存到 MLflow 中，以便進行後續的實驗追蹤和模型部署。
    """

    def __init__(self, logger: MLFlowLogger, **kwargs):
        """
        初始化 MLflowModelCheckpoint。

        Args:
            logger (MLFlowLogger): 用於連接 MLflow 的 logger 實例。
        """
        super().__init__(**kwargs)
        mlflow.set_experiment(experiment_id=logger.experiment_id)
        mlflow.start_run(run_id=logger.run_id)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """
        在訓練過程中保存模型檢查點到 MLflow。

        Args:
            trainer (pl.Trainer): PyTorch Lightning 訓練器。
            filepath (str): 要保存的模型檢查點文件的路徑。
        """
        mlflow.pytorch.log_model(trainer.model, "model")

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


# class MLflowModelCheckpoint(pl.Callback):
#     def __init__(self, monitor: str, mode: str, mlflow_logger: MLFlowLogger) -> None:
#         super().__init__()
#         self.monitor = monitor
#         self.mlflow_logger = mlflow_logger
#         self.mode = mode
#         self.best_metric = float("-inf") if mode == "max" else float("inf")
#         mlflow.set_tracking_uri(mlflow_logger._tracking_uri)
#         mlflow.set_experiment(experiment_id=mlflow_logger.experiment_id)
#         mlflow.start_run(run_id=mlflow_logger.run_id)

#     def on_validation_epoch_end(self, trainer, pl_module):
#         current_metric = trainer.callback_metrics[self.monitor].item()
#         if (self.mode == "min" and current_metric < self.best_metric) or (self.mode == "max" and current_metric > self.best_metric):
#             self.best_metric = current_metric
#             mlflow.pytorch.log_model(pl_module, "model")
