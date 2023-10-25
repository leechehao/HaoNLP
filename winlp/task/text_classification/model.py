from typing import Any, Type

import torch
import transformers
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from sklearn.metrics import classification_report

from winlp.core import Module

LABELS = "labels"


class TextClassificationModule(Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        monitor: str,
        mode: str,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSequenceClassification,
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, pretrained_model_name_or_path, num_labels, monitor, mode, **kwargs)

    def forward(self, batch: Any) -> None:
        return self.model(**batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        loss, logits = self.model(**batch)[:2]
        preds = torch.argmax(logits, dim=1)
        accuracy = classification_report(batch[LABELS].cpu(), preds.cpu(), output_dict=True, zero_division=0)["accuracy"]

        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.common_step("test", batch)
    