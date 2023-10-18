"""Chest CT 影像文字報告的實體識別資料集。"""


import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URLS = {
    "train": "data/train.conll",
    "validation": "data/validation.conll",
    "test": "data/test.conll",
}


class ChestCTNERConfig(datasets.BuilderConfig):
    """建立 ChestCTNER 設定。"""

    def __init__(self, **kwargs) -> None:
        super(ChestCTNERConfig, self).__init__(**kwargs)


class ChestCTNER(datasets.GeneratorBasedBuilder):
    """Chest CT 影像文字報告的實體識別資料集。"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = ChestCTNERConfig
    BUILDER_CONFIGS = [
        ChestCTNERConfig(name="chest_ct_ner", version=VERSION, description="Chest CT 影像文字報告的實體識別資料集。"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "B-Observation",
                            "I-Observation",
                            "B-Finding",
                            "I-Finding",
                            "B-Diagnosis",
                            "I-Diagnosis",
                            "B-Location",
                            "I-Location",
                            "B-Certainty",
                            "I-Certainty",
                            "B-Change",
                            "I-Change",
                            "B-Attribute",
                            "I-Attribute",
                            "B-Size",
                            "I-Size",
                        ],
                    ),
                ),
            },
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[-1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
