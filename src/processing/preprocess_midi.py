import os
import glob
import music21 as m21
import json
import numpy as np
import pandas as pd
import pathlib

current_path = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = current_path + "/../../data/raw/maestro-v3.0.0"
SAVE_PATH = current_path + "\\..\\..\\data\\processed\\maestro\\"
m21.environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')
m21.environment.set('midiPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')  # Atualize para o caminho correto do MuseScore
SEQUENCE_LENGTH = 64

ACCEPTABLE_DURATIONS = [
    0.25,   # sixteenth note 
    0.5,    # eighth note
    0.75,   # dotted eighth note (eighth note + sixteenth note)
    1.0,    # quarter note
    1.5,    # dotted quarter note (quarter note + dotted eighth note)
    2,      # half note
    3,      # 3 quarter notes
    4       # whole note
]


class ProcessingPipeline():

    def __init__(self, songs=[], acceptable_durations=[]) -> None:
        self.songs = songs
        self.songs_encoded = []
        self.acceptable_durations = acceptable_durations 
        self.song_dataset = None
        self.mappings = None


    def run(self, dataset_path, save_path, num_songs=16, num_measures=3):
        print("Loading songs...")
        self.load_songs(dataset_path, file_extension=(".midi", "mid"), song_limiter=num_songs, measures_limiter=num_measures)
        print(f"Loaded {len(self.songs)} songs.")

        for song_idx, song in enumerate(self.songs):
            # Filter out songs if wanted
            if not self.__has_acceptable_durations(song):
                continue

            # Transpose songs to Cmaj/Amin
            song_transposed = self.__transpose_song(song)

            # encode songs with music time series representation
            song_encoded = self.__encode_song(song_transposed, time_step=0.25)

            # save songs to text file
            self.save_song(song_encoded, save_path, song_idx)
            self.songs_encoded.append(song_encoded)

        self.map_symbols(save_path + "\\mapping.json")

    
    def save_song(self, song: m21.stream.Score, save_path, name_to_save):
        file_path = os.path.join(save_path, str(name_to_save))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file_path, "w") as fp:
            fp.write(song)


    def load_songs(self, dataset_path, file_extension=(".midi", "mid"), measures_limiter=16, song_limiter=None):
        """
        Can handle kern, midi, musicXML files
        """
        for path, subdir, files in os.walk(dataset_path):
            for file in files:
                
                if (len(self.songs) >= song_limiter):
                    break

                if file.endswith(file_extension):
                    print(file)
                    song = m21.converter.parse(os.path.join(path, file))
                    truncated_song = song.measures(0, measures_limiter) # Limit amount of measures per song
                    self.songs.append(truncated_song)
    

    def __has_acceptable_durations(self, song: m21.stream.Score):
        if self.acceptable_durations == []:
            return True

        for note in song.flatten().notesAndRests:
            if note.duration.quarterLength not in self.acceptable_durations:
                return False

        return True

    def __transpose_song(self, song: m21.stream.Score):
        key = song.analyze('key')
        
        # Transpose to C except if is minor key
        i = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
        if key.mode == "minor":
            i = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

        song_new = song.transpose(i)
        
        return song_new
    
    def __encode_song(self, song: m21.stream.Score, time_step):
        encoded_song_melody = []
        # parts = m21.instrument.partitionByInstrument(song)
        
        for element in song.flatten().notesAndRests:
            # Check if is a note or chord or rest
            symbol = None

            if isinstance(element, m21.note.Note):
                symbol = element.pitch.midi
            elif isinstance(element, m21.chord.Chord):
                symbol = '.'.join(str(n) for n in element.normalOrder)
            elif isinstance(element, m21.note.Rest):
                symbol = "r"

            steps = int(element.duration.quarterLength / time_step)
            for step in range(steps):
                if step == 0:
                    encoded_song_melody.append(symbol)
                    continue
                encoded_song_melody.append("_")

        encoded_song_melody = " ".join(map(str, encoded_song_melody))
        return encoded_song_melody 

    def create_dataset(self):
        df = pd.DataFrame(self.songs_encoded)
        return df

    def get_songs_as_int(self):
        int_songs = []

        for song in self.songs_encoded:
            song = song.split()
            int_song = []
            for symbol in song:
                int_song.append(self.mappings[symbol])
            int_songs.append(int_song)

        return int_songs

    def generating_training_sequences(self, sequence_length):
        int_songs = self.get_songs_as_int()
        print(int_songs)
        inputs = []
        targets = []
        print()
        # Generate training sequences
        for int_song in int_songs:
            num_sequences = len(int_song) - sequence_length -1
            print(len(int_song))
            print(num_sequences)
            sequence_inputs = []
            sequence_targets = []
            for i in range(num_sequences):
                sequence_inputs.append(int_song[i:i+sequence_length])
                sequence_targets.append(int_song[i+sequence_length])
            print(f"sequence {int_song}")
            print(sequence_inputs)
            inputs.append(sequence_inputs)
            targets.append(sequence_targets)
        
        print("sequence")
        print(inputs)
        print(targets)
        print(pd.DataFrame(inputs))
        print(pd.DataFrame(targets))
        # One-hot encode the sequences


        pass

    def map_symbols(self, mapping_path):
        """
        Creates a json file that maps the symbols in the song dataset onto integers
        """
        mappings = {}

        # Identify vocabulary
        songs_splited = [song.split() for song in self.songs_encoded]
        all_symbols = []
        for song in songs_splited:
            all_symbols.extend(song)
        
        vocabulary = list(set(all_symbols))
        vocabulary = sorted(set(item for item in all_symbols))
        
        # Create mappings
        for i, symbol in enumerate(vocabulary):
            mappings[symbol] = i
        
        # Save vocabulary to a json file
        with open(mapping_path, "w") as fp:
            json.dump(mappings, fp, indent=4)

        print(f"Mappings of notes:\n {mappings}")
        self.mappings = mappings


    def get_songs_original(self):
        return self.songs

    def get_songs_dataset(self):
        return self.songs_encoded

    def set_acceptable_durations(self, durations):
        self.acceptable_durations = durations



if __name__ == "__main__":
    p = ProcessingPipeline()
    p.run(DATASET_PATH, save_path=SAVE_PATH, num_songs=3, num_measures=16)
    # p.set_acceptable_durations(ACCEPTABLE_DURATIONS)

    # df = p.create_dataset()
    # print(df)
    songs = p.get_songs_original()
    # songs_encoded = p.get_songs_dataset
    # print(songs_encoded)
    song = songs[0]
    # print(song)
    p.generating_training_sequences(10)
    # song.show()
