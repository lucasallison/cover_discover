#!/usr/bin/env python3

import numpy as np
import os
import logging
import json
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, dataset_dict, Dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from model import add_special_tokens

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    """
    Loads data for causal language modeling.
    The data contains only raw text as an input which is duplicated
    for the reference (automatically shifted by one position).
    """

    def __init__(self, args, special_tokens, model_name=None):
        super().__init__()
        self.args = args
        self.model_name = model_name or self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if special_tokens:
            add_special_tokens(self.tokenizer, None, tokens=special_tokens)

        # Mac
        self.multiprocessing_context='fork' if torch.backends.mps.is_available() else None

    def setup(self, stage):
        data_dir = self.args.in_dir

        if stage == "fit":
            splits = ["train", "validation"]
        elif stage == "predict":
            splits = [self.args.split]

        raw_dataset = load_dataset(
            "json",
            data_files={split: os.path.join(data_dir, f"{split}.json") for split in splits},
            field="data",
        )

        self.dataset = self._process_raw_dataset(raw_dataset)

    def _process_raw_dataset(self, raw_dataset):
        dataset = {}

        for split in raw_dataset.keys():
            columns = ["attention_mask", "input_ids", "labels"]
            columns_to_remove = ["in"]

            dataset[split] = raw_dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=columns_to_remove,
            )

            dataset[split].set_format(type="torch", columns=columns)

        return dataset

    def _convert_to_features(self, example_batch, indices=None):
        features = self.tokenizer(example_batch["in"], max_length=self.args.max_length, truncation=True)
        features["labels"] = features["input_ids"]

        return features

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            num_workers=self.args.max_threads,
            collate_fn=self._pad_sequence,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.args.batch_size,
            num_workers=self.args.max_threads,
            collate_fn=self._pad_sequence,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.batch_size,
            num_workers=self.args.max_threads,
            collate_fn=self._pad_sequence,
        )

    def _pad_sequence(self, batch):
        """
        Add paddings to align sequence endings in a batch.
        """
        batch_collated = {}

        paddings = {
            # 0 is used as a dummy padding if pad_token_id is not available
            # (such as for GPT-2), attention mask still has to be set properly
            "input_ids": self.tokenizer.pad_token_id or 0,
            "attention_mask": 0,
            # -100 is a default `ignore_index` for torch.nn.NLLLoss
            "labels": -100,
        }
        for key in ["input_ids", "attention_mask", "labels"]:
            elems = [x[key] for x in batch]
            elems_pad = pad_sequence(elems, batch_first=True, padding_value=paddings[key])
            batch_collated[key] = elems_pad

        return batch_collated
