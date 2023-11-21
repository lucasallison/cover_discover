#!/usr/bin/env python3

import os
import argparse
import logging
import json
from sklearn.model_selection import train_test_split
import random

from collections import defaultdict, namedtuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPLITS = ["train", "validation", "test"]

class MidiDataset:
    def __init__(self, path, max_files=None):
        super().__init__()

        self.data = None 
        self.path = path

        # Maximum files that will be processed
        self.max_files = max_files

    def load(self, splits, path=None):
        block_size = 1024
        all_blocks = []
        processed_files = 0

        walk_dir = os.path.abspath(self.path)
        for r, _, files in os.walk(walk_dir):
            for f in files:
                path = f"{r}/{f}"

                logger.info('Processing = ' + path)
                with open(path, 'r') as midi_txt_file:
                    line = midi_txt_file.readline()
                    tokens = line.split(' ')
                    
                    while len(tokens) > 0:
                        block = tokens[0 : min(block_size, len(tokens))]
                        all_blocks.append(' '.join(block))
                        tokens = tokens[block_size:]

                processed_files += 1
                if self.max_files is not None and processed_files >= self.max_files:
                    break


        train, val_test = train_test_split(all_blocks, test_size=0.4)
        validation, test = train_test_split(val_test, test_size=0.5) 

        self.data = {"train": train, "validation": validation, "test": test}

class Preprocessor:
    """
    By default, a directory with processed dataset will contain the files `train.json`, `dev.json`, `test.json`,
    each file with the following structure:
    {
        "data" : [
            {... data entry #1 ...},
            {... data entry #2 ...},
            .
            .
            .
            {... data entry #N ...},
        ]
    }
    This format is expected for loading the data into PyTorch dataloaders for training and inference.
    """

    def __init__(self, dataset, out_dirname):
        self.dataset = dataset
        self.out_dirname = out_dirname

    def process(self, split):
        output = {"data": []}
        data = self.dataset.data[split]

        for entry in data:
            examples = []
            examples.append({"in": entry})

            for example in examples:
                output["data"].append(example)

        with open(os.path.join(self.out_dirname, f"{split}.json"), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=True, help="Base directory of the midi files converted to text")
    parser.add_argument("--output", type=str, required=True, help="Name of the output directory")
    parser.add_argument("--max_files", type=str, default=None, help="Maximum number of files to process")

    args = parser.parse_args()
    logger.info(args)
    dataset = MidiDataset(path=args.input)

    try:
        dataset.load(splits=SPLITS, path=args.output)
    except FileNotFoundError as err:
        logger.error(f"Dataset could not be loaded")
        raise err

    try:
        out_dirname = args.output
        os.makedirs(out_dirname, exist_ok=True)
    except OSError as err:
        logger.error(f"Output directory {out_dirname} can not be created")
        raise err

    preprocessor = Preprocessor(dataset=dataset, out_dirname=out_dirname)
    for split in SPLITS:
        preprocessor.process(split)

    logger.info(f"Preprocessing finished.")
