import os
import argparse
 
import music21
import pandas as pd
 

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("input_dir_name", help="name of input directory of mid files",
                    type=str)
	parser.add_argument("-o", "--output_dir_name", help="name of output directory of csv files, if argument not specified, will create one with suffix output-csvs",
					type=str)
					
	args = parser.parse_args()
	is_dir(args.input_dir_name)
	
	input_dir = os.fsencode(args.input_dir_name)
	output_dir_name = args.output_dir_name
	
	if not output_dir_name: 
		output_dir_name = args.input_dir_name + "-output-csvs"
		
	if not os.path.exists(output_dir_name):
		os.makedirs(output_dir_name)
		
	print("Outputting csv files in to " + output_dir_name)

	for file in os.listdir(input_dir):
		filename = os.fsdecode(file)
		print("Processing " + filename + " in to " + filename[:-4] + ".csv")
		if not (filename.endswith(".mid") or filename.endswith(".midi")):
			continue
		mf = music21.midi.MidiFile()
		mf.open(args.input_dir_name + "/" + filename)
		mf.read()
		mf.close()
		s = music21.midi.translate.midiFileToStream(mf, quantizePost=False).flat 
		df = pd.DataFrame(columns=["note_name", "start_time", "duration", "velocity", "tempo"])
		for g in s.recurse().notes:
			if g.isChord:
				for pitch in g.pitches: 
					x = music21.note.Note(pitch, duration=g.duration)
					x.volume.velocity = g.volume.velocity

					x.offset = g.offset
					s.insert(x)
		note_tempo = s.metronomeMarkBoundaries()[0][2].number
		for note in s.recurse().notes: 
			if note.isNote:
				new_df = pd.DataFrame([[note.pitch, round(float(note.offset), 3), round(note.duration.quarterLength, 3), note.volume.velocity, note_tempo]], columns=["note_name", "start_time", "duration", "velocity", "tempo"])
				df = pd.concat([df, new_df])

		df.to_csv(output_dir_name + "/" + filename[:-4] + ".csv")
		
	print("Done creating csvs!")


if __name__ == "__main__":
	main()
