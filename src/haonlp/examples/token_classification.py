import torch
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from haonlp.task.token_classification import TokenClassificationDataModule, TokenClassificationModule


torch.set_float32_matmul_precision("high")
L.seed_everything(42, workers=True)

datamodule = TokenClassificationDataModule(
    dataset_name="datasets_hub/token_classification/chest_ct_ner",
    pretrained_model_name_or_path="prajjwal1/bert-tiny",
    max_length=256,
    batch_size=4,
)

datamodule.setup(stage="fit")

model = TokenClassificationModule(
    pretrained_model_name_or_path="prajjwal1/bert-tiny",
    label_list=datamodule.label_list,
    monitor="val_f1",
    mode="max",
    learning_rate=1e-3,
)

mlf_logger = MLFlowLogger(
    experiment_name="token_classification",
    run_name="run_1",
    tracking_uri="/home/bryant/MyMLOps/exp",
)

trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    deterministic=True,
    max_epochs=10,
    logger=mlf_logger,
)
trainer.fit(model, datamodule)
