#import music21
#import ms3
import os
import sys
import time
from sys import platform

def help():
    print(f'Usage: python {sys.argv[0]} <project name> [midi file]', file=sys.stderr)
    print('If no MIDI file is given, the program starts with an empty score.', file=sys.stderr)
    sys.exit(1)

curr_dir = os.path.dirname(os.path.realpath(__file__))

mscore = 'mscore'
if platform == 'darwin':
    mscore = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'

if len(sys.argv) <= 1:
    print('Not enough arguments given.\n', file=sys.stderr)
    help()

project_name = sys.argv[1]
project_dir = f'{curr_dir}/{project_name}'
os.system(f'mkdir -p "{project_dir}"')

fiter = 0
def get_score_file():
    return f'{project_dir}/score_{fiter}.mscz'

if len(sys.argv) > 2:
    # Initial conversion of the input midi file to the first score file
    midi_file = sys.argv[2]
    convert(midi_file, get_score_file())
else:
    # If no MIDI file is given, start with empty project.
    os.system(f"cp {curr_dir}/empty.mscz {get_score_file()}")



def convert(inname, outname):
    os.system(f'env "{mscore}" {inname} -o {outname}')
def musescore_open(fname):
    os.system(f'env "{mscore}" {fname}')

def generate_new_midi():
    global fiter

    convert(get_score_file(), f'{project_dir}/model_input.mid')
    os.system(f"""
    {curr_dir}/../model/generate_mid_from_str.py \
        --experiment midi \
        --input "$(python3 {curr_dir}/../MIDI-LLM-tokenizer/midi_to_str.py {project_dir}/model_input.mid)" \
        --output {project_dir}/out_mid.txt \
        --max_length 200 \
        --beam_size 2 \
        --cpu \
        -v
    """)
    os.system(f'python3 {curr_dir}/../MIDI-LLM-tokenizer/str_to_midi.py "$(cat {project_dir}/out_mid.txt)" --output {project_dir}/model_output.mid')

    fiter += 1
    convert(f'{project_dir}/model_output.mid', get_score_file())


print('We are opening the score in MuseScore, only leave the parts that you want to be included in as inspiritation')
musescore_open(get_score_file())
#input('Press enter to continue: ')

while True:
    print('Generating MIDI based on input...')

    start = time.monotonic()
    generate_new_midi()
    end = time.monotonic() - start

    print(f'Generating MIDI took {end}s')

    print('We are opening the score in MuseScore, where we can repeat the cycle')
    musescore_open(get_score_file())

    res = input('Do you want to regenerate another pass?: ')
    if res.lower() not in {'y', 'yes'}:
        break

convert(get_score_file(), f'{project_dir}/output.mid')
