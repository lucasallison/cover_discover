#import music21
#import ms3
import os
import sys
import time

fiter = 0
def get_score_file():
    return f'score_{fiter}.mscz'

def convert(inname, outname):
    os.system(f"mscore {inname} -o {outname}")
def musescore_open(fname):
    os.system(f"mscore {fname}")

def generate_new_midi():
    global fiter

    convert(get_score_file(), 'model_input.mid')
    os.system(f"""
    ../generate_mid_from_str.py \
        --experiment midi \
        --input "$(python ../MIDI-LLM-tokenizer/midi_to_str.py test.mid)" \
        --output out_mid.txt \
        --max_length 200 \
        --beam_size 2 \
        --mps \
        -v
    """)
    os.system('python ../MIDI-LLM-tokenizer/str_to_midi.py "$(cat out_mid.txt)" --output model_output.mid')

    fiter += 1
    convert('model_output.mid', get_score_file())

# Initial conversion of the input midi file to the first score file
midi_file = sys.argv[1]
convert(midi_file, get_score_file())

print('We are opening the score in MuseScore, only leave the parts that you want to be included in as inspiritation')
musescore_open(get_score_file())
input('Press enter to continue: ')

while True:
    print('Generating MIDI based on input...')

    start = time.monotonic()
    generate_new_midi()
    end = time.monotonic() - start

    print(f'Generating MIDI took {end}s')

    print('We are opening the score in MuseScore, where we can repeat the cycle')
    musescore_open(get_score_file())

    res = input('Do you want to regenerate another pass?: ')
    if res not in {'y', 'Y', 'yes', 'YES'}:
        break
