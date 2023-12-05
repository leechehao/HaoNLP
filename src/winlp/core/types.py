from dataclasses import dataclass


MAX_LENGTH = "max_length"
TOKENS = "tokens"
TEXTS = "texts"
LABELS = "labels"
IGNORE_INDEX = -100


@dataclass
class SplitType:
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"


@dataclass
class MonitorModeType:
    MIN: str = "min"
    MAX: str = "max"
