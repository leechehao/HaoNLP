from typing import Optional, Sequence, Tuple, Type

import io
import importlib
from weakref import proxy

import onnx
import lightning as L
import transformers
import mlflow
import torch
from torch.optim import Optimizer, AdamW
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from winlp.core import types


class Module(L.LightningModule):
    """
    基於 PyTorch Lightning 的 Module，用於訓練和測試的深度學習模型模板。

    主要處理包括模型初始化、度量指標配置、優化器和學習速率調度器的配置，以及訓練期間的回調函數配置。
    """

    def __init__(
        self,
        downstream_model_type: Type[_BaseAutoModelClass],
        pretrained_model_name_or_path: str,
        label_list: list[str],
        monitor: str,
        mode: types.MonitorModeType,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.0,
    ) -> None:
        """
        初始化 Module。

        Args:
            downstream_model_type (Type[_BaseAutoModelClass]): 下游任務模型的類型。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_list (list[str]): 標籤名稱列表。
            monitor (str): 要監控的指標名稱。
            mode (types.MonitorModeType): 監控指標的模式，可選的值為 `min`、`max`。
            learning_rate (float, optional): 學習率。預設為 1e-3。
            weight_decay (float, optional): 權重衰減，用於正則化和防止模型過擬合。預設為 1e-2。
            warmup_ratio (float, optional): 預熱比例，在訓練初期學習率逐漸提升階段所佔的總訓練步驟比例。預設為 0.0。
        """
        super().__init__()
        self.save_hyperparameters()
        self.label_list = label_list
        self.model = downstream_model_type.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.monitor = monitor
        self.mode = mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self._hf_pipeline = None

    def setup(self, stage: str) -> None:
        """
        在每個階段開始時調用，用於配置模型相關的設置。

        Args:
            stage (str): 當前的訓練階段。
        """
        self.configure_metrics()

    def configure_metrics(self) -> None:
        """
        覆寫以配置用於訓練和驗證的度量指標。
        """
        pass

    @property
    def hf_pipeline_task(self) -> Optional[transformers.pipeline]:
        """
        覆寫以配置用於預測的 pipeline。
        """
        return None

    @property
    def num_labels(self) -> int:
        return len(self.label_list)

    @property
    def id2label(self) -> dict:
        return {i: tag for i, tag in enumerate(self.label_list)}

    @property
    def label2id(self) -> dict:
        return {tag: i for i, tag in enumerate(self.label_list)}

    @property
    def hf_pipeline(self):
        """
        覆寫以定義要使用的 Hugging Face pipeline。
        """
        if self._hf_pipeline is None:
            if self.hf_pipeline_task is not None:
                self._hf_pipeline = self.hf_pipeline_task
            else:
                raise NotImplementedError("此模型未定義任何 pipeline。")
        return self._hf_pipeline

    @hf_pipeline.setter
    def hf_pipeline(self, pipeline: transformers.pipeline):
        self._hf_pipeline = pipeline

    def hf_predict(self, inputs, *args, **kwargs):
        """
        Hugging Face pipeline，供預測使用。

        可輸入之額外參數 num_workers、batch_size。

        Args:
            inputs (_type_): 待預測的輸入。

        Returns:
            _type_: 預測結果。
        """
        self.eval()
        return self.hf_pipeline(inputs, *args, **kwargs)

    def configure_optimizers(self) -> Tuple[Sequence[Optimizer], Sequence[dict]]:
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

    def configure_callbacks(self) -> Sequence[L.Callback] | L.Callback:
        """
        配置訓練過程中使用的回調函數。

        Returns:
            Union[Sequence[L.Callback], L.Callback]: 一個或多個回調函數。
        """
        print(self.__class__.__module__)
        print(self.__class__.__name__)
        calback = MLflowModelCheckpoint(subclass_path=self.__class__.__module__, subclass_name=self.__class__.__name__)
        return [calback]

    def log_onnx_model(self) -> None:
        """
        將模型轉成 onnx 格式並上傳 MLflow

        Example:
            (1):
                ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                ort_session.run(None, dict(tokenizer(test_sentence, return_tensors="np")))
            (2):
                onnx_model = mlflow.onnx.load_model(onnx_model_uri)
                serialized_model = onnx_model.SerializeToString()
                ort_session = onnxruntime.InferenceSession(serialized_model)
                ort_session.run(None, dict(tokenizer(test_sentence, return_tensors="np")))
        """
        self.eval()
        onnx_byte_stream = io.BytesIO()

        input_sample = {"batch": dict(self.tokenizer("test sentence!", return_tensors="pt"))}
        dynamic_axes = {
            key: {0: "batch_size", 1: "sequence"} for key in input_sample["batch"]
        }
        self.to_onnx(
            onnx_byte_stream,
            input_sample=input_sample,
            export_params=True,
            input_names=list(input_sample["batch"].keys()),
            dynamic_axes=dynamic_axes,
        )

        onnx_byte_stream.seek(0)

        mlflow.onnx.log_model(onnx_model=onnx.load_model(onnx_byte_stream), artifact_path="onnx_model")


class MLflowModelCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    """
    專為 MLflow 整合的 ModelCheckpoint。

    主要功能是在訓練過程中選擇最佳模型檢查點，然後將該模型保存到 MLflow 中，以便進行後續的實驗追蹤和模型部署。
    """
    def __init__(self, subclass_path: str, subclass_name: str) -> None:
        super().__init__()
        self.subclass_path = subclass_path
        self.subclass_name = subclass_name

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """
        在訓練過程中保存模型檢查點到 MLflow。

        Args:
            trainer (L.Trainer): PyTorch Lightning 訓練器。
            filepath (str): 要保存的模型檢查點文件的路徑。
        """
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)
        # mlflow.pytorch.log_state_dict(checkpoint, "model")

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        model = getattr(importlib.import_module(self.subclass_path), self.subclass_name).load_from_checkpoint(buffer)
        mlflow.pytorch.log_model(model, "model")

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


# class MLflowModelCheckpoint(L.Callback):
#     def __init__(self, monitor: str, mode: str) -> None:
#         super().__init__()
#         self.monitor = monitor
#         self.mode = mode
#         self.best_metric = float("-inf") if mode == "max" else float("inf")

#     def on_validation_epoch_end(self, trainer, pl_module):
#         current_metric = trainer.callback_metrics[self.monitor].item()
#         if (self.mode == "min" and current_metric < self.best_metric) or (self.mode == "max" and current_metric > self.best_metric):
#             self.best_metric = current_metric
#             mlflow.pytorch.log_model(pl_module, "model")
