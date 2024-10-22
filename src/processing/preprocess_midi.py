import os
import music21 as m21

current_path = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = current_path + "/../../data/raw/maestro-v3.0.0/2018"
m21.environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')
m21.environment.set('midiPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')  # Atualize para o caminho correto do MuseScore
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
        self.acceptable_durations = acceptable_durations 


    def run(self, dataset_path):
        print("Loading songs...")
        self.load_songs(dataset_path)
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
            self.save_song(song_encoded, song_idx)
            self.songs[song_idx] = song_encoded 

    
    def save_song(self, song: m21.stream.Score, name_to_save):
        save_path = current_path + "\\..\\..\\data\\processed\\maestro\\"
        file_path = os.path.join(save_path, str(name_to_save))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file_path, "w") as fp:
            fp.write(song)


    def load_songs(self, dataset_path):
        """
        Can handle kern, midi, musicXML files
        """
        for path, subdir, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.midi', '.mid')):
                    print(file)
                    song = m21.converter.parse(os.path.join(path, file))
                    truncated_song = song.measures(0, 16)
                    self.songs.append(truncated_song)
                    
                    if (len(self.songs) > 3):
                        break
    

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
        
        encoded_song = []

        for event in song.flatten().notesAndRests:
            # Check if is a note or rest
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi

            elif isinstance(event, m21.note.Rest):
                symbol = "r"

            steps = int(event.duration.quarterLength / time_step)
            print(steps)
            for step in range(steps):
                if step == 0:
                    encoded_song.append(symbol)
                    continue
                encoded_song.append("_")

        encoded_song = " ".join(map(str, encoded_song))
        return encoded_song


    def get_songs(self):
        return self.songs

    def set_acceptable_durations(self, durations):
        self.acceptable_durations = durations


if __name__ == "__main__":
    p = ProcessingPipeline()
    p.run(DATASET_PATH)
#     print(DATASET_PATH)
#     p.load_songs(DATASET_PATH)
#     # p.set_acceptable_durations(ACCEPTABLE_DURATIONS)

    songs = p.get_songs()
    song = songs[0]
    print(songs)
#     for song in songs:
#         print(f"Has acceptable duration? {p.has_acceptable_durations(song)}")
#         if p.has_acceptable_durations(song)==True:
    # song.show()

# # MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--3.midi
# # MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--4.midi
# # MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1.midi