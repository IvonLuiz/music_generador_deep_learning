import os
import music21 as m21

current_path = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = current_path + "/../../data/raw/maestro-v3.0.0/2018"
m21.environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')
m21.environment.set('midiPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')  # Atualize para o caminho correto do MuseScore
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,    # quarter note
    1.5,    # dotted quarted note
]


class ProcessingPipeline():

    def __init__(self, songs=[], acceptable_durations=[]) -> None:
        self.songs = songs
        self.acceptable_durations = acceptable_durations 




    def load_songs(self, dataset_path):
        """
        Can handle kern, midi, musicXML files
        """
        for path, subdir, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.midi', '.mid')):
                    print(file)
                    song = m21.converter.parse(os.path.join(path, file))
                    truncated_song = song.measures(0, 10)
                    self.songs.append(truncated_song)
                    print(song)
                    self.__has_acceptable_durations(song)
                    
                    if (len(self.songs) > 5):
                        break
    

    def __has_acceptable_durations(self, song: m21.stream.Score):
        if self.acceptable_durations == []:
            return True

        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in self.acceptable_durations:
                return False

        return True


    def get_songs(self):
        return self.songs


    def set_acceptable_durations(self, durations):
        self.acceptable_durations = durations


if __name__ == "__main__":
    p = ProcessingPipeline()
    print(DATASET_PATH)
    p.load_songs(DATASET_PATH)
    songs = p.get_songs()
    
    song = songs[1]
    song.show()