import sys
from music21 import *

midi_file = sys.argv[1]

s = converter.parse(midi_file)
sp = midi.realtime.StreamPlayer(s)
sp.play()