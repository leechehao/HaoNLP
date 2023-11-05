"""IMDb 電影評論資料集。"""


from typing import List, Generator, Tuple, Dict

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

    def __init__(self, **kwargs) -> None:
        """
        初始化 IMDbReviewsConfig。

        Args:
            **kwargs: 可變關鍵字參數。
        """
        super(IMDbReviewsConfig, self).__init__(**kwargs)


class IMDb(datasets.GeneratorBasedBuilder):
    """
    IMDb 電影評論資料集。

    此資料集包含 IMDb 上的電影評論，以及對應的情感標籤（正面或負面）。

    Attributes:
        VERSION (datasets.Version): 資料集的版本。
        BUILDER_CONFIGS (list of IMDbReviewsConfig): 資料集的建構設定。
    """

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        IMDbReviewsConfig(name="imdb_sentiment_classification", version=VERSION, description="IMDb movie reviews dataset.")
    ]

    def _info(self) -> datasets.DatasetInfo:
        """
        返回資料集的基本資訊。

        Returns:
            datasets.DatasetInfo: 包含資料集的描述、引用、官網連結、授權、特徵和任務模板。
        """
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """
        定義資料集的分割產生器。

        Args:
            dl_manager (datasets.DownloadManager): 用於下載和解壓縮的管理器。

        Returns:
            List[datasets.SplitGenerator]: 資料集各個分割的生成器。
        """
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

    def _generate_examples(self, filepath) -> Generator[Tuple[int, Dict[str, int]], None, None]:
        """
        生成資料集範例。

        Args:
            filepath (str): 資料檔案的路徑。

        Yields:
            Tuple[int, Dict[str, int]]: 包含範例的 id 和內容的元組。
        """
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            labels_mapping = {"neg": 0, "pos": 1}
            for id_, row in enumerate(reader):
                if id_ == 0:
                    continue
                yield id_, {"texts": row[0], "labels": labels_mapping.get(row[1])}
