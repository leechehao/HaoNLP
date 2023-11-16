import unittest
from unittest.mock import patch, MagicMock

import datasets
import lightning as L
import transformers

from winlp.task.text_classification import TextClassificationDataModule, TextClassificationModule

TRAIN = "train"
TEXTS = "texts"


class TestTextClassificationDataModule(unittest.TestCase):

    @patch("transformers.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        self.data_module = TextClassificationDataModule(
            dataset_name="dummy_dataset",
            pretrained_model_name_or_path="dummy_model",
            label_column_name="dummy_column_name",
            max_length=512,
            batch_size=16,
        )

        mock_tokenizer.assert_called_once_with("dummy_model")

    def test_init(self):
        pass

    @patch("datasets.Dataset.map")
    @patch.object(TextClassificationDataModule, "_prepare_label_list")
    def test_process_data_train(self, mock_prepare_label_list, mock_map):
        self.data_module.process_data(datasets.Dataset, TRAIN)
        mock_map.call_args[0][0]({TEXTS: "dummy_text_1"})

        mock_prepare_label_list.assert_called_once()
        mock_map.assert_called_once()
        self.data_module.tokenizer.assert_called_once_with(
            "dummy_text_1",
            padding="max_length",
            truncation=True,
            max_length=self.data_module.max_length,
            return_tensors="pt",
        )

    @patch("datasets.Dataset.map")
    @patch.object(TextClassificationDataModule, "_prepare_label_list")
    def test_process_data_val_test(self, mock_prepare_label_list, mock_map):
        self.data_module.process_data(datasets.Dataset, None)
        mock_map.call_args[0][0]({TEXTS: "dummy_text_1"})

        mock_prepare_label_list.assert_not_called()
        mock_map.assert_called_once()
        self.data_module.tokenizer.assert_called_once_with(
            "dummy_text_1",
            padding=True,
            truncation=True,
            max_length=self.data_module.max_length,
            return_tensors="pt",
        )


class TestTextClassificationModule(unittest.TestCase):

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        self.model = TextClassificationModule(
            pretrained_model_name_or_path="dummy_model",
            label_list=["dummy_label_1", "dummy_label_2"],
            monitor="val_loss",
            mode="min",
        )

        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once_with("dummy_model")

    def test_init(self):
        self.assertEqual(self.model.best_metric, float("inf"))

    @patch("transformers.TextClassificationPipeline")
    def test_pipeline(self, mock_pipeline):
        self.model.hf_pipeline_task
        
        mock_pipeline.assert_called_once_with(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            truncation=True,
            device=self.model.device,
        )


class TestTextClassificationTrain(unittest.TestCase):

    def test_train(self):
        data_module = TextClassificationDataModule(
            dataset_name="/tmp2/ken/MyMLOps/datasets_hub/text_classification/imdb_sentiment_classification",
            pretrained_model_name_or_path="distilbert-base-uncased",
            label_column_name="labels",
        )
        data_module.setup("fit")

        model = TextClassificationModule(
            pretrained_model_name_or_path=data_module.pretrained_model_name_or_path,
            label_list=data_module.label_list,
            monitor="val_loss",
            mode="min",
        )

        trainer = L.Trainer(fast_dev_run=True)
        trainer.fit(model, data_module)


class TestTextClassificationPredict(unittest.TestCase):

    def test_predict(self):
        model = TextClassificationModule(
            pretrained_model_name_or_path="distilbert-base-uncased",
            label_list=["neg", "pos"],
            monitor="val_loss",
            mode="min",
        )

        model.hf_pipeline = transformers.TextClassificationPipeline(
            model=model.model,
            tokenizer=model.tokenizer,
            truncation=True,
            device=model.device,
        )

        y = model.hf_predict(["Good movie!"])
        self.assertEqual(len(y), 1)
        self.assertIn(y[0]["label"], model.label_list)
        self.assertIn("score", y[0])


if __name__ == "__main__":
    unittest.main()
