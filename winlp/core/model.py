from typing import Dict, Sequence, Tuple, Type

import lightning.pytorch as pl
import transformers
from torch.optim import Optimizer, AdamW
from transformers.models.auto.auto_factory import _BaseAutoModelClass


class Module(pl.LightningModule):
    def __init__(
        self,
        downstream_model_type: Type[_BaseAutoModelClass],
        pretrained_model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = downstream_model_type.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
        )
        self.num_labels = num_labels
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
