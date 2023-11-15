import unittest
from winlp.core import DataModule
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader


class TestDataModule(unittest.TestCase):
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("os.cpu_count")
    def setUp(self, mock_cpu_count, mock_tokenizer):
        mock_cpu_count.return_value = 8
        mock_tokenizer.return_value = MagicMock()
        self.data_module = DataModule(
            dataset_name="dummy_dataset",  # "datasets_hub/text_classification/imdb_sentiment_classification"
            pretrained_model_name_or_path="dummy_model", # "distilbert-base-uncased"
            label_column_name="dummy_column_name",
            max_length=512,
            batch_size=16
        )

        mock_cpu_count.assert_called_once()
        mock_tokenizer.assert_called_once_with("dummy_model")

    def test_init(self):
        self.assertEqual(self.data_module.dataset_name, "dummy_dataset") # "datasets_hub/text_classification/imdb_sentiment_classification"
        self.assertEqual(self.data_module.pretrained_model_name_or_path, "dummy_model")  # "distilbert-base-uncased"
        self.assertEqual(self.data_module.label_column_name, "dummy_column_name")
        self.assertEqual(self.data_module.max_length, 512)
        self.assertEqual(self.data_module.batch_size, 16)
        self.assertEqual(self.data_module.num_workers, 4)
        self.assertIsInstance(self.data_module.tokenizer, MagicMock)
        self.assertEqual(self.data_module._label_list, None)

    @patch("datasets.DatasetDict.set_format")
    @patch("winlp.core.DataModule.process_data")
    @patch("datasets.load_dataset")
    def test_setup(
        self,
        mock_dataset,
        mock_processed_data,
        mock_formatted_dataset_dict,
        stage = None,
    ):
        if stage == "fit":
            dummy_data = {
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            }
            mock_dataset.return_value = [Dataset.from_dict(dummy_data), Dataset.from_dict(dummy_data)]
            mock_processed_data.return_value = DatasetDict(
                {
                    "train": mock_dataset.return_value[0],
                    "validation": mock_dataset.return_value[1],
                }
            )
            mock_formatted_dataset_dict.return_value = mock_processed_data()
            self.data_module.setup(stage="fit")

            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("train", self.data_module.dataset)
            self.assertIn("validation", self.data_module.dataset)
        elif stage == "test":
            dummy_data = {
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            }
            mock_dataset.return_value = [Dataset.from_dict(dummy_data)]
            mock_processed_data.return_value = DatasetDict(
                {
                    "test": mock_dataset.return_value[0],
                }
            )
            mock_formatted_dataset_dict.return_value = mock_processed_data()
            self.data_module.setup(stage="test")

            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("test", self.data_module.dataset)
        else:
            dummy_data = {
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            }
            mock_dataset.return_value = DatasetDict(
                {
                    "train": Dataset.from_dict(dummy_data),
                    "validation": Dataset.from_dict(dummy_data),
                    "test": Dataset.from_dict(dummy_data),
                }
            )
            mock_formatted_dataset_dict.return_value = mock_processed_data()
            self.data_module.setup(stage=None)

            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("train", self.data_module.dataset)
            self.assertIn("validation", self.data_module.dataset)
            self.assertIn("test", self.data_module.dataset)

    def test_process_data(self):
        with self.assertRaises(NotImplementedError):
            self.data_module.process_data(MagicMock(), MagicMock())

    def test_train_dataloader(self):
        self.test_setup(stage="fit")
        train_dataloader = self.data_module.train_dataloader()

        self.assertIsInstance(train_dataloader, DataLoader)

    def test_val_dataloader(self):
        self.test_setup(stage="fit")
        val_dataloader = self.data_module.val_dataloader()

        self.assertIsInstance(val_dataloader, DataLoader)

    def test_test_dataloader(self):
        self.test_setup(stage="test")
        test_dataloader = self.data_module.test_dataloader()

        self.assertIsInstance(test_dataloader, DataLoader)

    @patch("winlp.core.DataModule.setup")
    def test_label_list(self, mock_setup):
        self.data_module.label_list

        mock_setup.assert_called_once_with("fit")

        self.data_module._label_list = ["dummy_label_1", "dummy_label_2"]
        
        self.assertEqual(self.data_module.label_list, self.data_module._label_list)

    @patch.object(DataModule, "label_list", new=["dummy_label_1", "dummy_label_2"])
    def test_num_labels(self):
        self.assertEqual(self.data_module.num_labels, 2)

if __name__ == "__main__":
    unittest.main()
