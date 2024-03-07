import unittest
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from haonlp.core import DataModule


class TestDataModule(unittest.TestCase):
    
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("os.cpu_count")
    def setUp(self, mock_cpu_count, mock_tokenizer):
        mock_cpu_count.return_value = 8
        mock_tokenizer.return_value = MagicMock()
        self.data_module = DataModule(
            dataset_name=["dummy_dataset_1", "dummy_dataset_2"],
            pretrained_model_name_or_path="dummy_model",
            label_column_name="dummy_column_name",
            max_length=512,
            batch_size=16
        )

        mock_cpu_count.assert_called_once()
        mock_tokenizer.assert_called_once_with("dummy_model")

    def test_init(self):
        self.assertEqual(self.data_module.dataset_name, ["dummy_dataset_1", "dummy_dataset_2"])
        self.assertEqual(self.data_module.pretrained_model_name_or_path, "dummy_model")
        self.assertEqual(self.data_module.label_column_name, "dummy_column_name")
        self.assertEqual(self.data_module.max_length, 512)
        self.assertEqual(self.data_module.batch_size, 16)
        self.assertEqual(self.data_module.num_workers, 4)
        self.assertIsInstance(self.data_module.tokenizer, MagicMock)
        self.assertEqual(self.data_module._label_list, None)

    @patch("datasets.DatasetDict.set_format")
    @patch("haonlp.core.DataModule.process_data")
    @patch("datasets.concatenate_datasets")
    @patch("datasets.load_dataset")
    def test_setup(
        self,
        mock_dataset,
        mock_concatenate_datasets,
        mock_processed_data,
        mock_formatted_dataset_dict,
        stage = None,
    ):
        if stage == "fit":
            dataset = Dataset.from_dict({
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            })
            mock_dataset.return_value = dataset
            mock_concatenate_datasets.return_value = dataset
            mock_processed_data.return_value = dataset
            mock_formatted_dataset_dict.return_value = DatasetDict(
                {
                    "train": dataset,
                    "validation": dataset,
                }
            )
            self.data_module.setup(stage="fit")

            self.assertEqual(mock_dataset.call_count, 4)
            self.assertEqual(mock_concatenate_datasets.call_count, 2)
            self.assertEqual(mock_processed_data.call_count, 2)
            mock_formatted_dataset_dict.assert_called_once()
            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("train", self.data_module.dataset)
            self.assertIn("validation", self.data_module.dataset)
        elif stage == "test":
            dataset = Dataset.from_dict({
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            })
            mock_dataset.return_value = dataset
            mock_concatenate_datasets.return_value = dataset
            mock_processed_data.return_value = dataset
            mock_formatted_dataset_dict.return_value = DatasetDict(
                {
                    "test": dataset,
                }
            )
            self.data_module.setup(stage="test")

            self.assertEqual(mock_dataset.call_count, 2)
            self.assertEqual(mock_concatenate_datasets.call_count, 1)
            self.assertEqual(mock_processed_data.call_count, 1)
            mock_formatted_dataset_dict.assert_called_once()
            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("test", self.data_module.dataset)
        else:
            dataset = Dataset.from_dict({
                "dummy_texts": ["dummy_text_1", "dummy_text_2", "dummy_text_3"],
                "dummy_labels": [1, 0, 1],
            })
            mock_dataset.return_value = dataset
            mock_concatenate_datasets.return_value = dataset
            mock_processed_data.return_value = dataset
            mock_formatted_dataset_dict.return_value = DatasetDict(
                {
                    "train": dataset,
                    "validation": dataset,
                    "test": dataset,
                }
            )
            self.data_module.setup(stage=None)

            self.assertEqual(mock_dataset.call_count, 6)
            self.assertEqual(mock_concatenate_datasets.call_count, 3)
            self.assertEqual(mock_processed_data.call_count, 3)
            mock_formatted_dataset_dict.assert_called_once()
            self.assertIsInstance(self.data_module.dataset, DatasetDict)
            self.assertIn("train", self.data_module.dataset)
            self.assertIn("validation", self.data_module.dataset)
            self.assertIn("test", self.data_module.dataset)

    def test_process_data(self):
        with self.assertRaises(NotImplementedError):
            self.data_module.process_data(MagicMock(), MagicMock())

    def test_train_dataloader(self):
        self.test_setup(stage="fit")

        self.assertIsInstance(self.data_module.train_dataloader(), DataLoader)

    def test_val_dataloader(self):
        self.test_setup(stage="fit")

        self.assertIsInstance(self.data_module.val_dataloader(), DataLoader)

    def test_test_dataloader(self):
        self.test_setup(stage="test")

        self.assertIsInstance(self.data_module.test_dataloader(), DataLoader)

    @patch("haonlp.core.DataModule.setup")
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
