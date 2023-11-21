# Finetune GPT-2 to produce midi songs 

- `midi_loop` - Convert midi files to text files
- `preprocess.py` - Preprocessing the midi text files to a dataset of JSON files.
- `dataloader.py`- Transforming preprocessed JSON datasets into PyTorch Dataloaders.
- `model.py` - Model-specific code for training.
- `inference.py` - Model-specific code for inference (decoding algorithms etc.).
- `train.py` - Running training.
- `interact.py` - Running interactive inference with user input.

### Finetuning GPT-2 on Midi files
Convert the midi archive to their text presresentation:
```
./midi_loop.sh
```
Process the midi txt files into train/validation/test datasets:
```
./preprocess.py \
    --input data/archive  \
    --output "data/data"
```
Fine tune GPT-2:
```
./train.py \
    --in_dir data/data \
    --experiment midi \
    --num_nodes 1 \
    --model_name gpt2 \
    --accumulate_grad_batches 4 \
    --learning_rate 5e-4 \
    --max_epochs 5
```

Models are saved in the file `out_dir/experiment/checkpoint_name` which can be set by the `--out_dir`, `--experiment` and `--checkpoint_name` flags. Only the experiment is required and by default it is stored under `experiments/experiment/model` 

To resume from a checkpoint: 
```
./train.py \ 
    --in_dir data/data \
    --experiment midi \
    --model_name gpt2 \
    --num_nodes 1 \
    --accumulate_grad_batches 4 \
    --learning_rate 5e-4 \
    --max_epochs 10 --model_path experiments/midi/model.ckpt --resume_training
```

Not that the amount of epochs that are run in the resumed training is max_epochs - trained epochs.


### Generating midi text using GPT-2

```
./interact.py \
    --experiment midi \
    --max_length 200 --mps
```
**Example output**
```
...
```