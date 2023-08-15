from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, AdamW


class Wav2Vec2WithProbe(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        tuning_type: str,
        lr: float,
        weight_decay: float,
        max_length: float,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._prepare_model(Wav2Vec2Model.from_pretrained(model_name, apply_spec_augment=False))
        self.fc = torch.nn.Linear(self.model.config.output_hidden_size, num_classes)
        self.max_feature_length = self.model._get_feat_extract_output_lengths(int(max_length * 16000))

        self._validation_outputs = []
        self._test_outputs = []

    @staticmethod
    def _grad(module, requires_grad):
        module.requires_grad = requires_grad
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _prepare_model(self, model):
        self._grad(model.feature_extractor, False)
        self._grad(model.feature_projection, False)
        self._grad(model.encoder, False)

        if self.hparams.tuning_type == "linear":
            model.encoder.layers = model.encoder.layers[:self.hparams.layer_index]
        elif self.hparams.tuning_type == "finetune":
            self._grad(model.encoder.layers[self.hparams.layer_index:], True)

        return model

    def _logits(self, hidden_states, mask):
        mask = self.model._get_feature_vector_attention_mask(self.max_feature_length, mask)
        mask = mask.unsqueeze(-1)
        avg_states = (hidden_states * mask).sum(1) / torch.sum(mask, dim=1)
        return self.fc(avg_states)

    def _log(self, split, loss, logits, labels):
        if not self.trainer.sanity_checking:
            self.log(f"{split}/loss", loss.item())
            self.log(f"{split}/acc", (logits.argmax(-1) == labels).float().mean().item())

    def _logits_labels(self, batch):
        inputs, labels = batch
        logits = self._logits(
            self.model(**inputs).last_hidden_state, inputs["attention_mask"])
        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self._logits_labels(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        self._log("train", loss, logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self._logits_labels(batch)
        self._validation_outputs.append({"logits": logits.detach(), "labels": labels.detach()})

    def test_step(self, batch, batch_idx):
        logits, labels = self._logits_labels(batch)
        self._test_outputs.append({"logits": logits.detach(), "labels": labels.detach()})

    @staticmethod
    def _accumulate_logits_labels(outputs):
        return torch.cat([o["logits"] for o in outputs]), torch.cat([o["labels"] for o in outputs])

    def on_validation_epoch_start(self):
        self._validation_outputs.clear()

    def on_validation_epoch_end(self):
        logits, labels = self._accumulate_logits_labels(self._validation_outputs)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        self._log("val", loss, logits, labels)

    def on_test_epoch_start(self):
        self._test_outputs.clear()

    def on_test_epoch_end(self):
        logits, labels = self._accumulate_logits_labels(self._test_outputs)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        self._log("test", loss, logits, labels)

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
