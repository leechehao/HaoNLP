from typing import Union

from functools import partial

import datasets
from transformers import PreTrainedTokenizerBase

from winlp.core import DataModule

TRAIN = "train"
MAX_LENGTH = "max_length"
TOKENS = "tokens"
NER_TAGS = "ner_tags"
LABELS = "labels"
IGNORE_INDEX = -100


class TokenClassificationDataModule(DataModule):
    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name_or_path: str,
        label_all_tokens: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dataset_name, pretrained_model_name_or_path, **kwargs)
        self.label_all_tokens = label_all_tokens

    def process_data(self, split_dataset: datasets.Dataset, split: str) -> datasets.Dataset:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=MAX_LENGTH if split == TRAIN else True,
            max_length=self.max_length,
            label_all_tokens=self.label_all_tokens,
        )
        split_dataset = split_dataset.map(convert_to_features, batched=True)
        return split_dataset

    @property
    def num_labels(self) -> int:
        return len(self.dataset[TRAIN].features[NER_TAGS].feature.names)

    @staticmethod
    def convert_to_features(
        examples,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[str, bool] = True,
        max_length: int = 512,
        label_all_tokens: bool = False,
    ):
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
