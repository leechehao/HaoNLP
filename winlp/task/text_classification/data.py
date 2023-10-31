import datasets

from winlp.core import DataModule


TRAIN = "train"
TEXTS = "texts"
LABELS = "labels"


class TextClassificationDataModule(DataModule):
    """
    文本分類資料模組。

    此模組提供了文本分類資料的處理功能，封裝了資料的預處理與獲取標籤數量的方法。

    Attributes:
        dataset_name (str): 資料集的名稱。遵從 Hugging Face dataset 格式。
        num_labels (int): 標籤數量。
        pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
        **kwargs: 其他關鍵字參數。

    Methods:
        process_data: 對指定的資料集進行預處理。
    """

    def __init__(
        self,
        dataset_name: str,
        num_labels: int,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> None:
        """
        初始化 TextClassificationDataModule。

        Args:
            dataset_name (str): 資料集的名稱。遵從 Hugging Face dataset 格式。
            num_labels (int): 標籤數量。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            **kwargs: 其他關鍵字參數。
        """
        super().__init__(dataset_name, num_labels, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs,)

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        """
        對指定的資料集進行預處理。

        使用 tokenizer 對文本資料進行編碼，並返回處理後的資料集。

        Args:
            split_dataset (datasets.Dataset): 要處理的資料集分割。
            split (str): 資料集的分割名稱（例如："train"）。

        Returns:
            datasets.Dataset: 處理後的資料集。
        """
        split_dataset = split_dataset.map(
            lambda example: self.tokenizer(
                example[TEXTS],
                padding="max_length" if split == TRAIN else True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers if split == TRAIN else None,
        )
        return split_dataset
