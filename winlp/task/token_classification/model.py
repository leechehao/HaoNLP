from typing import Any, Type

import torch
import transformers
import torchmetrics
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from winlp.core import Module


class TokenClassificationModule(Module):
    """
    Token Classification 任務的 PyTorch Lightning 模型模塊。
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        label_list: list[str],
        monitor: str,
        mode: str,
        label_all_tokens: bool = False,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForTokenClassification,
        **kwargs,
    ) -> None:
        """
        初始化 TokenClassificationModule。

        Args:
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            num_labels (int): 分類任務的標籤數量。
            label_list (list[str]): 標籤名稱列表。
            monitor (str): 要監控的指標名稱。
            mode (str): 監控指標的模式，例如 `min`、`max`。
            label_all_tokens (bool, optional): 若為 True，則對所有 subtoken（例如，tokenize 時）進行標記。預設為 False，即僅對第一個 subtoken 進行標記。
            downstream_model_type (Type[_BaseAutoModelClass], optional): 下游任務模型的類型。預設為 `transformers.  AutoModelForTokenClassification`
        """
        super().__init__(
            downstream_model_type,
            pretrained_model_name_or_path,
            num_labels,
            label_list,
            monitor,
            mode,
            **kwargs,
        )
        self.label_all_tokens = label_all_tokens
        self.best_metric = float("-inf") if mode == "max" else float("inf")

    def forward(self, batch: Any) -> transformers.modeling_outputs.TokenClassifierOutput:
        """
        前向傳播過程。

        Args:
            batch (Any): 輸入批次數據。

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput: 模型的輸出結果。
        """
        return self.model(**batch)

    def configure_metrics(self) -> None:
        """
        配置度量指標，用於訓練和驗證過程中的評估。
        """
        self.prec = torchmetrics.Precision(task="multiclass", num_classes=self.num_labels)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_labels)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_labels)
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc, "f1": self.f1}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        訓練過程中的一個訓練步驟。

        Args:
            batch (Any): 輸入批次數據。
            batch_idx (int): 批次索引。

        Returns:
            torch.Tensor: 訓練損失。
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        """
        驗證或測試過程中的共用步驟。

        Args:
            prefix (str): 步驟前綴，例如 `val` 或 `test`。
            batch (Any): 輸入批次數據。

        Returns:
            torch.Tensor: 驗證或測試損失。
        """
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        驗證過程中的一個驗證步驟。

        Args:
            batch (Any): 輸入批次數據。
            batch_idx (int): 批次索引。

        Returns:
            torch.Tensor: 驗證損失。
        """
        return self.common_step("val", batch)

    def on_validation_epoch_end(self):
        current_metric = self.trainer.callback_metrics[self.monitor].item()
        if (self.mode == "min" and current_metric < self.best_metric) or (self.mode == "max" and current_metric > self.best_metric):
            self.best_metric = current_metric
        self.log(f"best_{self.monitor}", self.best_metric)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        測試過程中的一個測試步驟。

        Args:
            batch (Any): 輸入批次數據。
            batch_idx (int): 批次索引。

        Returns:
            torch.Tensor: 測試損失。
        """
        return self.common_step("test", batch)

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor, mode="val") -> dict[str, torch.Tensor]:
        """
        計算度量指標。

        Args:
            predictions (torch.Tensor): 模型的預測結果。
            labels (torch.Tensor): 實際標籤。
            mode (str, optional): 計算模式。預設為 `val`.

        Returns:
            dict[str, torch.Tensor]: 包含度量指標結果的字典。
        """
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        return {f"{mode}_{k}": metric(predictions, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self):
        aggregation_strategy = "average" if self.label_all_tokens else "first"
        return transformers.TokenClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            device=self.device,
            aggregation_strategy=aggregation_strategy,
        )
