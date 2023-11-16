import unittest
from unittest.mock import patch, MagicMock

from transformers.models.auto.auto_factory import _BaseAutoModelClass
from torch.nn.parameter import Parameter
import torch
from lightning.pytorch import Trainer
from torch.optim import AdamW

from winlp.core import Module


class TestModule(unittest.TestCase):
    
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.models.auto.auto_factory._BaseAutoModelClass.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        self.model = Module(
            downstream_model_type=_BaseAutoModelClass,
            pretrained_model_name_or_path="dummy_model",
            label_list=["dummy_label_1", "dummy_label_2"],
            monitor="val_loss",
            mode="min",
            learning_rate=1e-3,
            weight_decay=1e-2,
            warmup_ratio=0.0,
        )

        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once_with("dummy_model")

    def test_init(self):
        self.assertEqual(self.model.label_list, ["dummy_label_1", "dummy_label_2"])
        self.assertIsInstance(self.model.model, MagicMock)
        self.assertIsInstance(self.model.tokenizer, MagicMock)
        self.assertEqual(self.model.monitor, "val_loss")
        self.assertEqual(self.model.mode, "min")
        self.assertEqual(self.model.learning_rate, 1e-3)
        self.assertEqual(self.model.weight_decay, 1e-2)
        self.assertEqual(self.model.warmup_ratio, 0.0)
        self.assertEqual(self.model._hf_pipeline, None)

    @patch("winlp.core.Module.configure_metrics")
    def test_setup(self, mock_configure_metrics):
        self.model.setup(stage=None)

        mock_configure_metrics.assert_called_once()

    def test_configure_metrics(self):
        try:
            self.model.configure_metrics()
        except Exception as e:
            self.fail(f"configure_metrics 方法出現異常 {e}")

    def test_hf_pipeline_task(self):
        try:
            self.assertEqual(self.model.hf_pipeline_task, None)
        except Exception as e:
            self.fail(f"hf_pipeline_task 方法出現異常 {e}")

    def test_num_labels(self):
        self.assertEqual(self.model.num_labels, 2)

    def test_id2label(self):
        self.assertEqual(self.model.id2label, {0: "dummy_label_1", 1: "dummy_label_2"})

    def test_label2id(self):
        self.assertEqual(self.model.label2id, {"dummy_label_1": 0, "dummy_label_2": 1})

    def test_hf_pipeline(self):
        with self.assertRaises(NotImplementedError):
            self.model.hf_pipeline

    def test_hf_pipeline_setter(self):
        self.model.hf_pipeline = MagicMock()
        
        self.assertIsInstance(self.model.hf_pipeline, MagicMock)

    @patch.object(Module, "hf_pipeline")
    @patch("lightning.pytorch.LightningModule.eval")
    def test_hf_predict(self, mock_eval, mock_hf_pipeline):
        self.model.hf_predict("dummy_inputs")

        mock_eval.assert_called_once()
        mock_hf_pipeline.assert_called_once_with("dummy_inputs")

    @patch("lightning.pytorch.LightningModule.parameters")
    def test_configure_optimizers(self, mock_parameters):
        mock_parameters.return_value = [Parameter(torch.randn(2, 2))]
        self.model.trainer = Trainer
        self.model.trainer.estimated_stepping_batches = 1
        optimizer, scheduler = self.model.configure_optimizers()

        self.assertIsInstance(optimizer[0], AdamW)
        self.assertEqual(scheduler[0], {"scheduler": self.model.scheduler, "interval": "step", "frequency": 1})

    @patch("winlp.core.model.MLflowModelCheckpoint")
    def test_configure_callbacks(self, mock_MLflowModelCheckpoint):
        self.model.configure_callbacks()

        mock_MLflowModelCheckpoint.assert_called_once_with(monitor=self.model.monitor, mode=self.model.mode)

    @patch("mlflow.onnx")
    @patch("onnx.load_model")
    @patch("lightning.pytorch.LightningModule.to_onnx")
    @patch("io.BytesIO")
    @patch("lightning.pytorch.LightningModule.eval")
    def test_log_onnx_model(self, mock_eval, mock_BytesIO, mock_to_onnx, mock_load_model, mock_log_model):
        mock_load_model.return_value = MagicMock()
        mock_log_model.log_model = MagicMock()
        self.model.log_onnx_model()

        mock_eval.assert_called_once()
        mock_BytesIO.assert_called_once()
        self.model.tokenizer.assert_called_once_with("test sentence!", return_tensors="pt")
        mock_to_onnx.assert_called_once()
        mock_BytesIO().seek.assert_called_with(0)
        mock_load_model.assert_called_once_with(mock_BytesIO())
        mock_log_model.log_model.assert_called_once_with(onnx_model=mock_load_model(), artifact_path="onnx_model")

if __name__ == "__main__":
    unittest.main()
