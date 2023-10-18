from typing import Optional

import os

import lightning.pytorch as pl
import datasets
import transformers
from torch.utils.data import DataLoader


TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
LABELS = "labels"


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name_or_path: str,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def setup(self, stage: str) -> None:
        dataset = datasets.load_dataset(path=self.dataset_name)
        for split in dataset:
            dataset[split] = self.process_data(dataset[split], split)

        dataset.set_format(type="torch", columns=self.tokenizer.model_input_names + [LABELS])
        self.dataset = dataset

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        return split_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[TRAIN], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[VALIDATION], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[TEST], batch_size=self.batch_size, num_workers=self.num_workers)
