from music21 import *


s = converter.parse('./test.midi')


note1 = note.Note("C4")
note1.duration.type = 'half'
note1.duration.quarterLength

s.append(note1)
s.repeatAppend(note1, 4)
s.show()

sp = midi.realtime.StreamPlayer(s)
sp.play()

