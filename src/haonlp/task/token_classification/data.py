from typing import Any, Union

from functools import partial

import datasets
import transformers
from transformers import PreTrainedTokenizerBase

from haonlp.core import DataModule, types


class TokenClassificationDataModule(DataModule):
    """
    用於處理 Token Classification 任務的數據集模組。
    """

    def __init__(
        self,
        dataset_name: list[str],
        pretrained_model_name_or_path: str,
        label_column_name: str,
        label_all_tokens: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化 TokenClassificationDataModule。

        Args:
            dataset_name (list[str]): 數據集的名稱。遵從 Hugging Face dataset 格式。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_column_name (str): 資料集裡標籤的欄位名稱。
            label_all_tokens (bool, optional): 是否標記所有的 tokens。預設為 False。
        """
        super().__init__(dataset_name, pretrained_model_name_or_path, label_column_name, **kwargs)
        self.label_all_tokens = label_all_tokens

    def process_data(self, split_dataset: datasets.Dataset, split: types.SplitType) -> datasets.Dataset:
        """
        預處理數據集，轉換成模型可用的格式。

        Args:
            split_dataset (datasets.Dataset): 要處理的數據集分割。
            split (types.SplitType): 數據集的分割名稱。

        Returns:
            datasets.Dataset: 轉換後的數據集分割。
        """
        if split == types.SplitType.TRAIN:
            self._prepare_label_list(split_dataset)

        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            label_column_name=self.label_column_name,
            padding=types.MAX_LENGTH if split == types.SplitType.TRAIN else True,
            max_length=self.max_length,
            label_all_tokens=self.label_all_tokens,
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
        padding: Union[str, bool] = True,
        max_length: int = 512,
        label_all_tokens: bool = False,
    ) -> transformers.BatchEncoding:
        """
        將示例轉換為模型輸入特徵。

        Args:
            examples (Any): 要轉換的示例。
            tokenizer (PreTrainedTokenizerBase): 預訓練模型的 tokenizer。
            label_column_name (str): 資料集裡標籤的欄位名稱。
            padding (Union[str, bool], optional): 是否填充到最大長度。預設為 True。
            max_length (int, optional): 最大序列長度。預設為 512。
            label_all_tokens (bool, optional): 是否標記所有 tokens。預設為 False。

        Returns:
            transformers.BatchEncoding: 包含轉換後特徵的 BatchEncoding 對象。
        """
        tokenized_inputs = tokenizer(
            examples[types.TOKENS],
            padding=padding,
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(types.IGNORE_INDEX)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else types.IGNORE_INDEX)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs[types.LABELS] = labels
        return tokenized_inputs

    def _prepare_label_list(self, dataset: datasets.Dataset) -> None:
        self._label_list = dataset.features[self.label_column_name].feature.names
