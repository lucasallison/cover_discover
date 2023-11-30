#!/usr/bin/env python3

import logging
import torch

from model import TrainingModule

from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

class InferenceModule:
    def __init__(self, args, model_path=None):
        self.args = args
        self.beam_size = args.beam_size
        self.special_tokens = TrainingModule.special_tokens

        if model_path is not None:
            self.model = TrainingModule.load_from_checkpoint(model_path)
            self.model.freeze()
            self.model_name = self.model.model.name_or_path
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model_name = args.model_name
            self.model = TrainingModule(args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def predict(self, s):
        inputs = self.tokenizer(s, return_tensors="pt")

        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            self.model.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        elif hasattr(self.args, "mps") and self.args.mps:
            mps_device = torch.device("mps")
            self.model.to(mps_device)
            return self.generate(inputs["input_ids"].detach().clone().to(mps_device))
        elif hasattr(self.args, "cpu") and self.args.cpu:
            cpu_device = torch.device("cpu")
            self.model.to(cpu_device)
            return self.generate(inputs["input_ids"].detach().clone().to(cpu_device))
        else:
            logger.warning("GPU, MPS, or CPU not specified")

        return self.generate(inputs["input_ids"])

    def generate(self, input_ids):
        # top-k sampling, other methods TBD
        out = self.model.model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            max_length=self.args.max_length,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        sentences = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return sentences
