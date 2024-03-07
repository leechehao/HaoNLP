from typing import Any, Union

from functools import partial

import datasets
import transformers
from transformers import PreTrainedTokenizerBase

from haonlp.core import DataModule, types


class BoundaryDetectionDataModule(DataModule):
    """
    用於處理 Boundary Detection 任務的數據集模組。
    """

    def __init__(
        self,
        dataset_name: list[str],
        pretrained_model_name_or_path: str,
        label_column_name: str,
        indices_column_name: str,
        label_list: list[str],
        **kwargs,
    ) -> None:
        """
        初始化 BoundaryDetectionDataModule。

        Args:
            dataset_name (list[str]): 數據集的名稱。遵從 Hugging Face dataset 格式。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_column_name (str): 資料集裡標籤的欄位名稱。
            indices_column_name (str): 資料集裡紀錄界線索引資訊的欄位名稱。
            label_list (list[str]): 資料集的標籤列表。
        """
        super().__init__(dataset_name, pretrained_model_name_or_path, label_column_name, **kwargs)
        self.indices_column_name = indices_column_name
        self._label_list = label_list
        self.label2id = {tag: i for i, tag in enumerate(self._label_list)}

    def process_data(self, split_dataset: datasets.Dataset, split: types.SplitType) -> datasets.Dataset:
        """
        預處理數據集，轉換成模型可用的格式。

        Args:
            split_dataset (datasets.Dataset): 要處理的數據集分割。
            split (types.SplitType): 數據集的分割名稱。

        Returns:
            datasets.Dataset: 轉換後的數據集分割。
        """
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            label_column_name=self.label_column_name,
            indices_column_name=self.indices_column_name,
            label2id=self.label2id,
            padding=types.MAX_LENGTH if split == types.SplitType.TRAIN else True,
            max_length=self.max_length,
        )
        split_dataset = split_dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.num_workers if split == types.SplitType.TRAIN else None,
        )
        return split_dataset

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        label_column_name: str,
        indices_column_name: str,
        label2id: dict[str, int],
        padding: Union[str, bool] = True,
        max_length: int = 512,
    ) -> transformers.BatchEncoding:
        """
        將示例轉換為模型輸入特徵。

        Args:
            examples (Any): 要轉換的示例。
            tokenizer (PreTrainedTokenizerBase): 預訓練模型的 tokenizer。
            label_column_name (str): 資料集裡標籤的欄位名稱。
            indices_column_name (str): 資料集裡紀錄界線索引資訊的欄位名稱。
            label2id (dict[str, int]): 標籤對應數值的字典。
            padding (Union[str, bool], optional): 是否填充到最大長度。預設為 True。
            max_length (int, optional): 最大序列長度。預設為 512。

        Returns:
            transformers.BatchEncoding: 包含轉換後特徵的 BatchEncoding 對象。
        """
        tokenized_inputs = tokenizer(
            examples["Text"],
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        labels = []
        for i, (indices, tags) in enumerate(zip(examples[indices_column_name], examples[label_column_name])):
            label_ids = [0 if val == 1 else types.IGNORE_INDEX for val in tokenized_inputs["attention_mask"][i]]
            indices = eval(indices)
            tags = eval(tags)
            for j, tag_index in enumerate(indices):
                for m, offset in enumerate(tokenized_inputs["offset_mapping"][i]):
                    if offset[0] <= tag_index and tag_index < offset[1]:
                        label_ids[m] = label2id[tags[j]]
            labels.append(label_ids)
        tokenized_inputs[types.LABELS] = labels
        return tokenized_inputs
