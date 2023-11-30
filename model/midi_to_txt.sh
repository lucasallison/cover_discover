#!/usr/bin/env bash

export LC_CTYPE=C
export LANG=C

cd ../MIDI-LLM-tokenizer/

for f in $(find ../model/data/archive -name '*.mid'); do
    out="$(echo $f | sed 's;/data/archive/;/data/output/;')"
    mkdir -p $(dirname "$out")
    echo "Processing $f"
    python3 ./midi_to_str.py "$f" --output "${out%.mid}.txt"
done
