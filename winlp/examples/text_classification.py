import os

from datasets import load_dataset
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

from winlp.task.text_classification import TextClassificationDataModule, TextClassificationModule


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
pl.seed_everything(0, workers=True)

mlf_logger = MLFlowLogger(
    experiment_name="Text-Classification-IMDb", 
    run_name="distilbert-base-uncased-1", 
    tracking_uri="./test/mlflow", 
    log_model=True, 
)

dataset_name = "datasets/text_classification/imdb_sentiment_classification"

datamodule = TextClassificationDataModule(
    dataset_name=dataset_name, 
    pretrained_model_name_or_path="distilbert-base-uncased", 
    batch_size=32, 
)
# datamodule.setup(stage="fit")

model = TextClassificationModule(
    pretrained_model_name_or_path="distilbert-base-uncased", 
    num_labels=load_dataset(dataset_name)["train"].features["labels"].num_classes,   # datamodule.num_labels, 
    learning_rate=2e-5, 
    )

trainer = pl.Trainer(
    accelerator="auto", 
    deterministic=True, 
    max_epochs=5, 
    default_root_dir="./test/pytorch_lightning", 
    logger=mlf_logger, 
    # num_sanity_val_steps=0,
)

trainer.fit(model, datamodule=datamodule)

trainer.test(datamodule=datamodule)
