import torch
import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.loggers import MLFlowLogger

from winlp.task.text_classification import TextClassificationDataModule, TextClassificationModule


# 設置環境變量和PyTorch設置
torch.set_float32_matmul_precision("high")
pl.seed_everything(0, workers=True)


# 設置資料集
dataset_name = "datasets/text_classification/imdb_sentiment_classification"

# 初始化資料模組
datamodule = TextClassificationDataModule(
    dataset_name=dataset_name,
    pretrained_model_name_or_path="distilbert-base-uncased",
    max_length=512,
    batch_size=32,
)
# datamodule.setup(stage="fit")

# 初始化文本分類模型
model = TextClassificationModule(
    pretrained_model_name_or_path="distilbert-base-uncased",
    num_labels=load_dataset(dataset_name)["train"].features["labels"].num_classes,  # datamodule.num_labels, 
    monitor="val_loss",
    mode="min",
    learning_rate=2e-5,
)

# 設置 MLFlow 紀錄器
mlf_logger = MLFlowLogger(
    experiment_name="Text-Classification-IMDb",
    run_name="distilbert-base-uncased-1",
    tracking_uri="./test/mlflow",
    log_model=False,
)

# 設置訓練器
trainer = pl.Trainer(
    accelerator="auto",
    deterministic=True,
    max_epochs=5,
    logger=mlf_logger,
)

# 開始訓練模型
trainer.fit(model, datamodule)

# 測試
# trainer.test(model, datamodule, "best")
