import os
import music21 as m21

current_path = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = current_path + "/../../data/raw/maestro-v3.0.0/2018"
m21.environment.set('musicxmlPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')
m21.environment.set('midiPath', 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe')  # Atualize para o caminho correto do MuseScore


class ProcessingPipeline():

    def __init__(self) -> None:
        self.songs = []


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

                    # if (len(self.songs) > 5):
                    #     break

    def preprocess(dataset_path):

        print("Loading songs...")

        pass
    
    def get_songs(self):
        return self.songs


if __name__ == "__main__":
    p = ProcessingPipeline()
    print(DATASET_PATH)
    p.load_songs(DATASET_PATH)
    songs = p.get_songs()
    
    print(f"Loaded {len(songs)} songs.")
    song = songs[1]
    song.show()