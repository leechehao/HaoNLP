import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

from winlp.task.text_classification.data import TextClassificationDataModule
from winlp.task.text_classification.model import TextClassificationModule


mlf_logger = MLFlowLogger(
    experiment_name="Text-Classification-IMDb", 
    run_name="distilbert-base-uncased-1", 
    tracking_uri="./mlflow", 
    log_model=True,
)

dataset = TextClassificationDataModule(dataset_name="imdb", pretrained_model_name_or_path="distilbert-base-uncased")
model = TextClassificationModule()
trainer = pl.Trainer(
    accelerator="auto", 
    max_epochs=5, 
    default_root_dir="./pytorch_lightning", 
    logger=mlf_logger, 
    num_sanity_val_steps=0,
)

trainer.fit(model, datamodule=dataset)

trainer.test(datamodule=dataset)
