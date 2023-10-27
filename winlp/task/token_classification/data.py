from typing import Any, Union

from functools import partial

import datasets
import transformers
from transformers import PreTrainedTokenizerBase

from winlp.core import DataModule

TRAIN = "train"
MAX_LENGTH = "max_length"
TOKENS = "tokens"
NER_TAGS = "ner_tags"
LABELS = "labels"
IGNORE_INDEX = -100


class TokenClassificationDataModule(DataModule):
    """
    用於處理 Token Classification 任務的數據集模組。
    """

    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name_or_path: str,
        label_all_tokens: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化 TokenClassificationDataModule。

        Args:
            dataset_name (str): 數據集的名稱。遵從 Hugging Face dataset 格式。
            pretrained_model_name_or_path (str): 預訓練模型的名稱或路徑。
            label_all_tokens (bool, optional): 是否標記所有的 tokens。預設為 False。
        """
        super().__init__(dataset_name, pretrained_model_name_or_path, **kwargs)
        self.label_all_tokens = label_all_tokens

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        """
        預處理數據集，轉換成模型可用的格式。

        Args:
            split_dataset (datasets.Dataset): 要處理的數據集分割。
            split (str): 數據集的分割名稱。

        Returns:
            datasets.Dataset: 轉換後的數據集分割。
        """
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=MAX_LENGTH if split == TRAIN else True,
            max_length=self.max_length,
            label_all_tokens=self.label_all_tokens,
        )
        split_dataset = split_dataset.map(convert_to_features, batched=True, num_proc=self.num_workers if split == "train" else None)
        return split_dataset

    @property
    def num_labels(self) -> int:
        """
        返回標籤的數量。

        Returns:
            int: 標籤的數量。
        """
        return len(self.dataset[TRAIN].features[NER_TAGS].feature.names)

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[str, bool] = True,
        max_length: int = 512,
        label_all_tokens: bool = False,
    ) -> transformers.BatchEncoding:
        """
        將示例轉換為模型輸入特徵。

        Args:
            examples (Any): 要轉換的示例。
            tokenizer (PreTrainedTokenizerBase): 預訓練模型的 tokenizer。
            padding (Union[str, bool], optional): 是否填充到最大長度。預設為 True。
            max_length (int, optional): 最大序列長度。預設為 512。
            label_all_tokens (bool, optional): 是否標記所有 tokens。預設為 False。

        Returns:
            transformers.BatchEncoding: 包含轉換後特徵的 BatchEncoding 對象。
        """
        tokenized_inputs = tokenizer(
            examples[TOKENS],
            padding=padding,
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[NER_TAGS]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(IGNORE_INDEX)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else IGNORE_INDEX)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs[LABELS] = labels
        return tokenized_inputs
