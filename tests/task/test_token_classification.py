import unittest
from unittest.mock import patch

import datasets
import transformers
import lightning as L

from haonlp.task.token_classification import TokenClassificationDataModule, TokenClassificationModule

TOKENS = "tokens"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
INPUT_IDS = "input_ids"
LABELS = "labels"


class TestTokenClassificationDataModule(unittest.TestCase):

    def setUp(self):
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenzier:
            self.data_module = TokenClassificationDataModule(
                dataset_name=["dummy_dataset_name"],
                pretrained_model_name_or_path="dummy_pretrained_model_name_or_path",
                label_column_name="dummy_label_column_name",
            )
            mock_tokenzier.assert_called_once_with(self.data_module.pretrained_model_name_or_path)

        self.label_list = ["POS", "NEG"]
        tokens = [["hello", "world"]]
        tags = [[0, 1]]
        features = datasets.Features({
            TOKENS: datasets.Sequence(datasets.Value("string")),
            self.data_module.label_column_name: datasets.Sequence(datasets.features.ClassLabel(names=self.label_list)),
        })
        data_dict = {TOKENS: tokens, self.data_module.label_column_name: tags}
        self.split_dataset = datasets.Dataset.from_dict(data_dict, features=features)

        self.data_module.tokenizer.return_value = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")(
            tokens,
            padding="max_length",
            truncation=True,
            max_length=5,
            is_split_into_words=True,
        )

    def test_init(self):
        self.assertEqual(self.data_module.label_all_tokens, False)

    def test_process_data_train(self):
        processed_dataset = self.data_module.process_data(self.split_dataset, TRAIN)
        self.assertIn(LABELS, processed_dataset.features)
        self.assertEqual(len(processed_dataset[0][INPUT_IDS]), len(processed_dataset[0][LABELS]))
        self.assertEqual(self.data_module._label_list, self.label_list)

    def test_process_data_validation(self):
        processed_dataset = self.data_module.process_data(self.split_dataset, VALIDATION)
        self.assertIn(LABELS, processed_dataset.features)
        self.assertEqual(len(processed_dataset[0][INPUT_IDS]), len(processed_dataset[0][LABELS]))
        self.assertEqual(self.data_module._label_list, None)

    def test_process_data_test(self):
        processed_dataset = self.data_module.process_data(self.split_dataset, TEST)
        self.assertIn(LABELS, processed_dataset.features)
        self.assertEqual(len(processed_dataset[0][INPUT_IDS]), len(processed_dataset[0][LABELS]))
        self.assertEqual(self.data_module._label_list, None)


class TestTokenClassificationModule(unittest.TestCase):

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForTokenClassification.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        dummy_pretrained_model_name_or_path = "dummy_pretrained_model_name_or_path"
        self.mode = "max"
        self.model = TokenClassificationModule(
            pretrained_model_name_or_path=dummy_pretrained_model_name_or_path,
            label_list=["POS", "NEG"],
            monitor="dummy_monitor",
            mode=self.mode,
        )
        mock_model.assert_called_once_with(
            dummy_pretrained_model_name_or_path,
            num_labels=self.model.num_labels,
            id2label=self.model.id2label,
            label2id=self.model.label2id,
        )
        mock_tokenizer.assert_called_once_with(dummy_pretrained_model_name_or_path)

    def test_init(self):
        self.assertEqual(self.model.label_all_tokens, False)
        self.assertEqual(self.model.best_metric, float("-inf"))

    @patch("transformers.TokenClassificationPipeline")
    def test_pipeline(self, mock_pipeline):
        self.model.hf_pipeline_task
        mock_pipeline.assert_called_once_with(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            device=self.model.device,
            aggregation_strategy="first",
        )


def test_train():
    pretrained_model_name_or_path = "prajjwal1/bert-tiny"
    data_module = TokenClassificationDataModule(
        dataset_name=["/home/bryant/MyMLOps/datasets_hub/token_classification/chest_ct_ner"],
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        label_column_name="ner_tags",
    )
    data_module.setup("fit")

    model = TokenClassificationModule(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        label_list=data_module.label_list,
        monitor="val_f1",
        mode="max",
    )

    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, data_module)


def test_predict():
    model = TokenClassificationModule(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        label_list=["Pos", "Neg"],
        monitor="val_f1",
        mode="max",
    )

    model.hf_pipeline = transformers.TokenClassificationPipeline(
        model=model.model,
        tokenizer=model.tokenizer,
        device=model.device,
        aggregation_strategy=None,
    )

    y = model.hf_predict("Have a good day!")
    assert len(y) == 5
    assert [a["word"] for a in y] == ["have", "a", "good", "day", "!"]
    for ent in [a["entity"] for a in y]:
        assert ent in ["Pos", "Neg"]
