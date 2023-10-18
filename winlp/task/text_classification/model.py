from winlp.core.model import *

# class TextClassificationModule():
#     def __init__(self, base_model, num_labels):
#         super().__init__()
#         self.model = DistilBertForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
#         self.train_loss = []
#         self.val_loss = []
#         self.val_preds = []
#         self.val_labels = []
#         self.test_preds = []
#         self.test_labels = []

#     def setup(self, _config, stage=None):
#         self.logger.log_hyperparams(_config)

#     def forward(self, input_ids, attention_mask):
#         return self.model(input_ids, attention_mask)

#     def configure_optimizers(self, lr):
#         return torch.optim.AdamW(self.model.parameters(), lr)

#     def training_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["label"]
#         logits = self.model(input_ids, attention_mask).logits

#         train_loss = torch.nn.functional.cross_entropy(logits, labels)
#         self.train_loss.append(train_loss)

#         self.log("train_loss", train_loss.item(), on_epoch=True)

#         return train_loss
    
#     def on_train_epoch_end(self, _run):
#         self.logger.log_metrics({"train_loss_epoch_mc": torch.stack(self.train_loss).mean().item()}, self.current_epoch)
#         # self.logger.experiment.log_metric(run_id=mlf_logger.run_id, 
#         #                                   key="train_loss_epoch_mc", 
#         #                                   value=torch.stack(self.train_loss).mean().item(), 
#         #                                   step=self.current_epoch)
#         _run.log_scalar("train_loss_epoch_mc", torch.stack(self.train_loss).mean().item())
#         self.train_loss.clear()

#     def validation_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["label"]
#         logits = self.model(input_ids, attention_mask).logits

#         val_loss = torch.nn.functional.cross_entropy(logits, labels)
#         self.val_loss.append(val_loss)

#         self.log("val_loss", val_loss.item(), on_epoch=True)

#         predictions = torch.argmax(logits, dim=1)
#         self.val_preds.extend(predictions.cpu())
#         self.val_labels.extend(labels.cpu())

#     def on_validation_epoch_end(self, _run):
#         self.logger.log_metrics({"val_loss_epoch_mc": torch.stack(self.val_loss).mean().item()}, self.current_epoch)
#         # self.logger.experiment.log_metric(run_id=mlf_logger.run_id, 
#         #                                   key="val_loss_epoch_mc", 
#         #                                   value=torch.stack(self.val_loss).mean().item(), 
#         #                                   step=self.current_epoch)
#         _run.log_scalar("val_loss_epoch_mc", torch.stack(self.val_loss).mean().item())
#         self.val_loss.clear()

#         self.logger.log_metrics({"val_acc_epoch": accuracy_score(self.val_preds, self.val_labels)}, self.current_epoch)
#         # self.logger.experiment.log_metric(run_id=mlf_logger.run_id, 
#         #                                   key="val_acc_epoch", 
#         #                                   value=accuracy_score(self.val_preds, self.val_labels), 
#         #                                   step=self.current_epoch)
#         _run.log_scalar("val_acc_epoch", accuracy_score(self.val_preds, self.val_labels))
#         self.val_preds.clear()
#         self.val_labels.clear()

#     def test_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["label"]
#         logits = self.model(input_ids, attention_mask).logits

#         predictions = torch.argmax(logits, dim=1)
#         self.test_preds.extend(predictions.cpu())
#         self.test_labels.extend(labels.cpu())

#     def on_test_epoch_end(self, _run):
#         self.logger.log_metrics({"test_acc_epoch": accuracy_score(self.test_preds, self.test_labels)}, self.current_epoch)
#         # self.logger.experiment.log_metric(run_id=mlf_logger.run_id, 
#         #                                   key="test_acc_epoch", 
#         #                                   value=accuracy_score(self.test_preds, self.test_labels), 
#         #                                   step=self.current_epoch)
#         _run.log_scalar("test_acc_epoch", accuracy_score(self.test_preds, self.test_labels))