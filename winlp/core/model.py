from typing import Dict, Sequence, Tuple, Type, Union

from weakref import proxy

import lightning.pytorch as pl
import transformers
import mlflow
from torch.optim import Optimizer, AdamW
from transformers.models.auto.auto_factory import _BaseAutoModelClass


class Module(pl.LightningModule):
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
        # self.logger.log_hyperparams()  # 待補 hydra model_config
        self.configure_metrics()

    def configure_metrics(self) -> None:
        pass

    def configure_optimizers(self) -> Tuple[Sequence[Optimizer], Sequence[Dict]]:
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.trainer.estimated_stepping_batches * self.warmup_ratio,
        )
        scheduler = {"scheduler": self.scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [scheduler]

    def configure_callbacks(self) -> Union[Sequence[pl.callbacks.Callback], pl.callbacks.Callback]:
        calback = MLflowModelCheckpoint(logger=self.logger, monitor=self.monitor, mode=self.mode)
        return [calback]


class MLflowModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        mlflow.set_experiment(experiment_id=logger.experiment_id)
        mlflow.start_run(run_id=logger.run_id)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
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
