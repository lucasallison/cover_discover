#!/bin/sh

pip install -r requirements.txt
curl -L 'https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021370-x86_64.AppImage' > ./app/mscore && chmod +x ./app/mscore

mkdir -p ./model/experiments/midi/
curl -L 'https://lieuwe.xyz/f/model.ckpt' > ./model/experiments/midi/model.ckpt
