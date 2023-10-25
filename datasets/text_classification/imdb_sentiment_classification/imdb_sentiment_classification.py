"""IMDb movie reviews dataset."""


import csv

import datasets
from datasets.tasks import TextClassification


logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = ""
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""

_URLS = {
    "train": "data/train.csv",
    "validation": "data/validation.csv",
    "test": "data/test.csv",
}


class IMDbReviewsConfig(datasets.BuilderConfig):
    """建立 IMDbReviews 設定。"""

    def __init__(self, **kwargs):
        super(IMDbReviewsConfig, self).__init__(**kwargs)


class IMDb(datasets.GeneratorBasedBuilder):
    """IMDb movie reviews dataset."""
    
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        IMDbReviewsConfig(name="imdb_sentiment_classification", version=VERSION, description="IMDb movie reviews dataset.")
    ]

    def _info(self):
        features = datasets.Features(
            {
                "texts": datasets.Value("string"),
                "labels": datasets.ClassLabel(names=["neg", "pos"]),
            },
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
            task_templates=[TextClassification(text_column="texts", label_column="labels")],
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"],},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"],},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"],},
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            labels_mapping = {"neg": 0, "pos": 1}
            for id_, row in enumerate(reader):
                if id_ == 0:
                    continue
                yield id_, {"texts": row[0], "labels": labels_mapping.get(row[1])}
