import datasets

from winlp.core import DataModule, types


class TextClassificationDataModule(DataModule):
    """
    文本分類資料模組。

    此模組提供了文本分類資料的處理功能，封裝了資料的預處理與獲取標籤數量的方法。

    Attributes:
        dataset_name (list[str]): 資料集的名稱。遵從 Hugging Face dataset 格式。
        pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
        **kwargs: 其他關鍵字參數。

    Methods:
        process_data: 對指定的資料集進行預處理。
    """

    def __init__(
        self,
        dataset_name: list[str],
        pretrained_model_name_or_path: str,
        label_column_name: str,
        **kwargs,
    ) -> None:
        """
        初始化 TextClassificationDataModule。

        Args:
            dataset_name (list[str]): 資料集的名稱。遵從 Hugging Face dataset 格式。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_column_name (str): 資料集裡標籤的欄位名稱。
            **kwargs: 其他關鍵字參數。
        """
        super().__init__(dataset_name, pretrained_model_name_or_path, label_column_name, **kwargs)

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
        split_dataset = split_dataset.class_encode_column(self.label_column_name)

        if split == types.SplitType.TRAIN:
            self._prepare_label_list(split_dataset)

        split_dataset = split_dataset.map(
            lambda example: self.tokenizer(
                example[types.TEXTS],
                padding=types.MAX_LENGTH if split == types.SplitType.TRAIN else True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers if split == types.SplitType.TRAIN else None,
        )
        return split_dataset

    def _prepare_label_list(self, dataset: datasets.Dataset) -> None:
        self._label_list = dataset.features[self.label_column_name].names
