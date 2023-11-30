# Audio Processing and Indexing Final Project

This is the repository that contains our code for the final project of the API course. Here we explored the possibility of generating a midi cover using transformers.

## Cover Discover: Running the app

Before running the app make sure the pip requirements are installed (`pip install -r requirements.txt`) and musescore (`sudo apt-get install musescore`). Then it can be run with `python app/main.py`.  

For Mac and Linux users: run `python app/play_midi.py my_song.mid` to play the generated midi file.


## Fine-tune GPT-2 to produce midi songs 

The following files in the `model` directory can be used to fine-tune and interact with a transformer. We choose to use GPT-2.

- `midi_to_txt` - Convert midi files to txt files
- `preprocess.py` - Preprocessing the midi txt files to a dataset of JSON files.
- `dataloader.py`- Transform preprocessed JSON datasets into PyTorch Dataloaders.
- `model.py` - Contains the Lightning training module.
- `inference.py` - Code for inference (generating text)
- `train.py` - Train the model.
- `interact.py` - Interact with the model. This mostly for testing. 
- `generate_mid_form_str.py` - Produces a valid midi txt string from a given input.


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

Note that the amount of epochs that are run in the resumed training is max_epochs - trained epochs.

### Generate a 'cover' from an input midi string

```
./generate_mid_from_str.py \
    --experiment midi \
    --input "<start> ... <end>" \
    --output out_mid.txt \
    --max_length 200 \
    --beam_size 2 \
    --mps \
    -v 
```

### Interacting with the (fine-tuned) GPT-2 model

To test the model interactively the following command can be used:

```
./interact.py \
    --experiment midi \
    --max_length 200 --mps
```
