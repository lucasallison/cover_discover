#!/usr/bin/env python

import argparse
import sys
import logging
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl

from inference import InferenceModule
from model import TrainingModule

from pprint import pprint as pp

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Checks if the input is a valid midi string
# If full_song is False
def is_valid_midi_str(input, full_song=False):

    tokens = input.split(' ')
    if (full_song and 
        not (re.match(r'^<start>$', tokens[0]) 
        and re.match(r'^<end>$', tokens[-1]))):
        logger.exception(f'Full midi song must start with <start> and end with <end>')
        return False
    
    # If we are not checking full songs we still want
    # inputs that start with <start> en end with <end>
    # to be considered as valid
    if not full_song and re.match(r'^<start>$', tokens[0]):
        tokens = tokens[1:]

    if not full_song and re.match(r'^<end>$', tokens[-1]):
        tokens = tokens[:-1]

    for token in tokens:
        # note:velocity:instrument or wait (t\d)
        if not (re.match(r'^[a-z]+:\d+[a-z]*:[a-z0-9]$', token) or 
        re.match(r'^t\d+$', token)):
            logger.exception(f'Invalid token: {token}')
            return False
    return True
    

# Remove all invalid tokens and ensure that it start with <start> 
# and ends with <end>
def clean_midi_str(input):

    tokens = input.split(' ')
    final_tokens = ['<start>']

    for token in tokens:

        # We only add valid tokens
        if (re.match(r'^[a-z]+:\d+[a-z]*:[a-z0-9]$', token) or 
        re.match(r'^t\d+$', token)):
            final_tokens.append(token)
        
    final_tokens.append('<end>')
    return ' '.join(final_tokens)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        default="experiments",
        type=str,
        help="Base directory of the experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Name of the experiment directory from which the model is loaded if `--model_name` is not specified.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the pretrained model used for prediction if `--experiment` is not specified.",
    )

    parser.add_argument("--max_threads", default=8, type=int, help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=5, type=int, help="Beam size.")
    parser.add_argument("--gpus", default=0, type=int, help="Number of GPUs.")
    parser.add_argument("--mps", action='store_true', help="Use MPS.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum number of tokens per example",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit precision. Packages `bitsandbytes` and `accelerate` need to be installed.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The midi text string that will be used as inpsiration to generate a new midi file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The path to the output midi file",
    )
    parser.add_argument(
        "-v",
        action='store_true',
        help="verbose output",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.max_threads)

    if args.experiment is not None and args.model_name is not None:
        raise ValueError(
            "The parameters `experiment` and `model_name` are mutually exclusive,\
            please specify only one of these."
        )

    elif args.experiment is None and args.model_name is None:
        raise ValueError("Please specify one of the following parameters: `experiment` OR `model_name`")


    if args.experiment:
        model_path = os.path.join(curr_dir, args.exp_dir, args.experiment, args.checkpoint)
        midi_model = InferenceModule(args, model_path=model_path)

    elif args.model_name:
        midi_model = InferenceModule(args)

    if not is_valid_midi_str(args.input):
        logger.exception("Program terminated due to invalid input.")
        sys.exit(1)

    logger.info("Generating cover...")
    midi_str = midi_model.predict(args.input)[0]

    if args.v:
        logger.info(f"Generated output: {midi_str}")

    midi_str = clean_midi_str(midi_str)
    f = open(args.output, "w")
    f.write(midi_str)
    logger.info(f"Cover generated! Written to {args.output}")


