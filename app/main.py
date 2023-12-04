#import music21
#import ms3
import os
import sys
import time
from sys import platform
import argparse
import hashlib

def file_hash(fname):
    with open(fname, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


fiter = 0

def main(args):
    global fiter

    def help():
        args.print_help()
        sys.exit(1)

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    mscore = 'mscore'
    if platform == 'darwin':
        mscore = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'

    project_dir = f'{curr_dir}/{args.project_name}'
    os.system(f'mkdir -p "{project_dir}"')

    def get_score_file():
        return f'{project_dir}/score_{fiter}.mscz'

    if args.midi_file is not None:
        # Initial conversion of the input midi file to the first score file
        convert(args.midi_file, get_score_file())
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

        exp_dir = ''
        if args.exp_dir is not None:
            exp_dir = f'--exp_dir "{args.exp_dir}"'
        experiment = ''
        if args.experiment is not None:
            experiment = f'--experiment "{args.experiment}"'
        model_name = ''
        if args.model_name is not None:
            model_name = f'--model_name "{args.model_name}"'

        os.system(f"""
        {curr_dir}/../model/generate_mid_from_str.py \
            {exp_dir} \
            {experiment} \
            {model_name} \
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

        os.system(f'cp "{get_score_file()}" "{get_score_file()}.bak"')

        hash_before = file_hash(get_score_file())
        musescore_open(get_score_file())
        hash_after = file_hash(get_score_file())

        if hash_before == hash_after:
            break

    convert(get_score_file(), f'{project_dir}/output.mid')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CoverDiscover',
        description='Generate covers based on MIDI input',
    )
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

    parser.add_argument(
        'project_name',
        type=str,
        help='Name of the cover project',
    )
    parser.add_argument(
        'midi_file',
        type=str,
        nargs='?',
        default=None,
        help='MIDI file to use as a starting point. If no MIDI file is given, the program starts with an empty score.',
    )

    args = parser.parse_args()

    if args.experiment is not None and args.model_name is not None:
        raise ValueError(
            "The parameters `experiment` and `model_name` are mutually exclusive,\
            please specify only one of these."
        )

    elif args.experiment is None and args.model_name is None:
        raise ValueError("Please specify one of the following parameters: `experiment` OR `model_name`")

    main(args)
