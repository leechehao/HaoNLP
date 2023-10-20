import datasets

from winlp.core import DataModule

TRAIN = "train"
PADDING = "max_length"
TEXTS = "texts"
LABELS = "labels"


class TextClassificationDataModule(DataModule):
    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> None:
        super().__init__(dataset_name, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs,)

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        split_dataset = split_dataset.map(
            lambda example: self.tokenizer(
                example[TEXTS],
                padding=PADDING,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=self.batch_size,
        )
        return split_dataset

    @property
    def num_labels(self) -> int:
        return self.dataset[TRAIN].features[LABELS].num_classes
