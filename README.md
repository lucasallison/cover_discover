# Training GPT-2 on the Tiny Shakspeare dataset

- `preprocess.py` - Preprocessing the tiny Shakespeare dataset into JSON files.
- `dataloader.py`- Transforming preprocessed JSON datasets into PyTorch Dataloaders.
- `model.py` - Model-specific code for training.
- `inference.py` - Model-specific code for inference (decoding algorithms etc.).
- `train.py` - Running training.
- `decode.py` - Running batch inference on dev / test splits.
- `interact.py` - Running interactive inference with user input.


## Examples: Language Modeling

### Interactive Prompting with (pre-traind) GPT-2
```
./interact.py \
    --model_name gpt2  \
    --max_length 200
```

```
[In]: Good morning!
[Out]:
['Good morning! I hope you all enjoyed my breakfast in the café. We are '
 'working with the media and I will provide more updates about the situation '
 'in your home and our working plans after the election, as well as an update '
 'on where our progress is going. My name is Richard Coughlin and I was in '
 'your office for our first business meeting.\n' (...)]
```

### Finetuning GPT-2 on Tiny Shakespeare
```
./preprocess.py \
    --dataset "tiny_shakespeare.txt"  \
    --dataset_dir "data/orig/" \
    --output "data/tiny_shakespeare"
```
```
./train.py \
    --in_dir data/tiny_shakespeare \
    --experiment tiny_shakespeare \
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
    --in_dir data/tiny_shakespeare \
    --experiment tiny_shakespeare-2 \
    --model_name gpt2 \
    --num_nodes 1 \
    --accumulate_grad_batches 4 \
    --learning_rate 5e-4 \
    --max_epochs 10 --model_path experiments/tiny_shakespeare/model.ckpt --resume_training
```

Not that the amount of epochs that are run in the resumed training is max_epochs - trained epochs.


### Generating Shakespeare using GPT-2

./interact.py \
    --experiment tiny_shakespeare \
    --max_length 200 --mps
```
**Example output**
```
[In]: Good morning! 
[Out]:
['Good morning! \n'
 '\n'
 'PETRUCHIO:\n'
 'And thou shalt have a father till she speak,\n'
 "For my son's sake be ready,\n"
 "I charge thee, be thou ready at five o'clock.\n" (...)]
```