from typing import Any, Type

import torch
import transformers
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from sklearn.metrics import classification_report

from winlp.core import Module, types


class TextClassificationModule(Module):
    """
    文本分類模組。

    此模組用於建立一個文本分類任務的 PyTorch Lightning 模型模塊。它包裝了模型初始化、前向傳播、訓練步驟、驗證步驟和測試步驟。

    Attributes:
        pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
        label_list (list[str]): 標籤名稱列表。
        monitor (str): 監控指標的名稱。
        mode (types.MonitorModeType): 監控模式，如 "min" 或 "max"。
        downstream_model_type (Type[_BaseAutoModelClass]): 預設使用的模型類型。
        **kwargs: 其他關鍵字參數。

    Methods:
        forward: 定義模型的前向傳播過程。
        training_step: 定義訓練過程中的單次訓練步驟。
        common_step: 定義一個通用的步驟，用於計算損失和準確度。
        validation_step: 定義驗證過程中的單次驗證步驟。
        test_step: 定義測試過程中的單次測試步驟。
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        label_list: list[str],
        monitor: str,
        mode: types.MonitorModeType,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSequenceClassification,
        **kwargs,
    ) -> None:
        """
        初始化 TextClassificationModule。

        Args:
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_list (list[str]): 標籤名稱列表。
            monitor (str): 監控指標的名稱。
            mode (types.MonitorModeType): 監控模式，如 "min" 或 "max"。
            downstream_model_type (Type[_BaseAutoModelClass], optional): 使用的模型類型，預設為 AutoModelForSequenceClassification。
            **kwargs: 其他關鍵字參數。
        """
        super().__init__(
            downstream_model_type,
            pretrained_model_name_or_path,
            label_list,
            monitor,
            mode,
            **kwargs,
        )
        self.best_metric = float("-inf") if mode == types.MonitorModeType.MAX else float("inf")

    def forward(self, batch: Any) -> transformers.modeling_outputs.SequenceClassifierOutput:
        """
        定義模型的前向傳播過程。

        Args:
            batch (Any): 輸入到模型的批次資料。

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: 模型的輸出結果。
        """
        return self.model(**batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        定義訓練步驟，並計算損失。

        Args:
            batch (Any): 輸入到模型的批次資料。
            batch_idx (int): 批次索引。

        Returns:
            torch.Tensor: 計算得到的損失值。
        """
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def common_step(self, prefix: str, batch: Any) -> None:
        """
        定義驗證和測試通用的步驟，用於計算損失和準確度。

        Args:
            prefix (str): 計算指標的前綴，如 "val" 或 "test"。
            batch (Any): 輸入到模型的批次資料。
        """
        loss, logits = self.model(**batch)[:2]
        preds = torch.argmax(logits, dim=1)
        accuracy = classification_report(batch[types.LABELS].cpu(), preds.cpu(), output_dict=True, zero_division=0)["accuracy"]

        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        定義驗證步驟。

        Args:
            batch (Any): 輸入到模型的批次資料。
            batch_idx (int): 批次索引。
        """
        self.common_step("val", batch)

    def on_validation_epoch_end(self):
        current_metric = self.trainer.callback_metrics[self.monitor].item()
        if (self.mode == types.MonitorModeType.MIN and current_metric < self.best_metric) or (self.mode == types.MonitorModeType.MAX and current_metric > self.best_metric):
            self.best_metric = current_metric
        self.log(f"best_{self.monitor}", self.best_metric)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        定義測試步驟。

        Args:
            batch (Any): 輸入到模型的批次資料。
            batch_idx (int): 批次索引。
        """
        self.common_step("test", batch)

    @property
    def hf_pipeline_task(self):
        return transformers.TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            device=self.device,
        )
