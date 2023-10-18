import lightning.pytorch as pl

from winlp.task.token_classification import TokenClassificationDataModule, TokenClassificationModule


pl.seed_everything(42, workers=True)

datamodule = TokenClassificationDataModule(
    dataset_name="chest_ct_ner",
    pretrained_model_name_or_path="prajjwal1/bert-tiny",
    max_length=256,
    batch_size=4,
)

datamodule.setup(stage="fit")

model = TokenClassificationModule(
    pretrained_model_name_or_path="prajjwal1/bert-tiny",
    num_labels=datamodule.num_labels,
    learning_rate=1e-3,
)

trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    deterministic=True,
    max_epochs=10,
)
trainer.fit(model, datamodule)
