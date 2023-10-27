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
    """
    基於 PyTorch Lightning 的 DataModule，用於處理自然語言處理（NLP）任務數據集的模板。

    封裝了從 Hugging Face 的 `datasets` 和 `transformers` 庫加載並預處理數據集的過程，使其能夠
    方便地與 PyTorch 模型集成。
    """

    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name_or_path: str,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ) -> None:
        """
        初始化 DataModule。

        Args:
            dataset_name (str): 數據集的名稱。遵從 Hugging Face dataset 格式。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            max_length (int, optional): 在 tokenization 過程中序列的最大長度。預設為 512。
            batch_size (int, optional): DataLoader 中每批次加載的數據樣本數量。預設為 16。
            num_workers (Optional[int], optional): DataLoader 用於加載數據的工作進程數。若為 None，則自動設置為機器 CPU 核心數的一半。預設為 None。
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def setup(self, stage: str) -> None:
        """
        加載和預處理數據集。

        Args:
            stage (str): `fit`、`validate`、`test` 或 `predict`。
        """
        dataset = datasets.load_dataset(path=self.dataset_name)
        for split in dataset:
            dataset[split] = self.process_data(dataset[split], split)

        dataset.set_format(type="torch", columns=self.tokenizer.model_input_names + [LABELS])
        self.dataset = dataset

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        """
        預處理數據集。這個方法應在子類中實現，以進行數據集的清洗、過濾、轉換等操作。

        Args:
            split_dataset (datasets.Dataset): 要進行預處理的數據集分割，例如 `train`、`validation` 或 `test`。
            split (str): 指定當前正在處理的數據集分割部分，如 `train`、`validation` 或 `test`。

        Raises:
            NotImplementedError: 如果該方法未在子類中實現，則拋出此異常。

        Returns:
            datasets.Dataset: 預處理後的數據集分割。此數據集準備好被模型訓練或測試使用。
        """
        raise NotImplementedError("必須在子類中實現數據集的預處理流程。")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[TRAIN], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[VALIDATION], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[TEST], batch_size=self.batch_size)
