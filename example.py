from music21 import *


file = 'test'
# file = 'Bohemian_Rhapsody'

score = converter.parse(file + '.midi')

# Print the initial score
score.show('t')
print('--------')

# Change the F4 note to a C4 note in the first chord
# We use recure so we access all subelements of the score
first_chord = score.recurse().getElementsByClass(chord.Chord)[0]
first_chord.remove(note.Note('F4'))
first_chord.add(note.Note('C4'))

# Add the first three notes of the first chord to the score
for i, n in enumerate(first_chord):
    if i > 2:
        break
    score.insertIntoNoteOrChord(float(i + 1), n)

# Transpose up 2 semitones 
for p in score.parts[0].pitches:
    p.transpose(2, inPlace=True)

# Create notest C4, C5, D5 and append them to the score. 
# The score is one bar so the 
# notes appear on the fourth and fifth beats 
score.append(note.Note('C4'))
score.append(note.Note('C5'))
score.append(note.Note('D5'))

# Add the E4 and G4 notes to the fourth beat to 
# create a full C chord
c = chord.Chord('E4 G4') 
score.insertIntoNoteOrChord(4.0, c)


# Flatten the added layers into a single layer
score = score.flatten()

# Print and play the resulting score
score.show('t')
sp = midi.realtime.StreamPlayer(score)
sp.play()

# Write to midi file
score.write('midi', fp=file + '_cover.midi')
