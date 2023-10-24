from typing import Any, Dict, Sequence, Type, Union

import torch
import transformers
import lightning.pytorch as pl
import torchmetrics
import mlflow
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from winlp.core import Module


class TokenClassificationModule(Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        monitor: str,
        mode: str,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForTokenClassification,
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, pretrained_model_name_or_path, num_labels, monitor, mode, **kwargs)

    def forward(self, batch):
        return self.model(**batch)

    def configure_metrics(self) -> None:
        self.prec = torchmetrics.Precision(task="multiclass", num_classes=self.num_labels)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_labels)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels)
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc, "f1": self.f1}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def compute_metrics(self, predictions, labels, mode="val") -> Dict[str, torch.Tensor]:
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        return {f"{mode}_{k}": metric(predictions, labels) for k, metric in self.metrics.items()}
