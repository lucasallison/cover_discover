#!/usr/bin/env python3

import logging
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import (
    AdamW,
)

logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer, model, tokens):
    """
    Allows to add custom special tokens to the model vocabulary.
    """
    special_tokens_dict = {"additional_special_tokens": tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


class TrainingModule(pl.LightningModule):
    special_tokens = []

    def __init__(self, args, **kwargs):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, return_dict=True)
        self.args = args
        self.save_hyperparameters(ignore=["datamodule"])
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        self.datamodule = kwargs.get("datamodule", None)

    def forward(self, **inputs):
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return {"loss": out["loss"], "logits": out["logits"]}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log("loss/val", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )
        max_steps = -1 if self.args.max_steps is None else self.args.max_steps
        max_epochs = int(self.args.max_epochs)

        total_steps = max_epochs * len(self.datamodule.train_dataloader()) if max_steps == -1 else max_steps
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "polynomial",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(f"Using Adam optimizer")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # these should be reasonable defaults for the seq2seq models
        # for finetuning GPT-2,you can use higher learning rate (e.g., 5e-4)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser
