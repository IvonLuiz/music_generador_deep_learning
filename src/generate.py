import os
import pickle
from pathlib import Path
import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from modeling.vae import VAE
from modeling.train import SPECTROGRAMS_PATH, load_fsdd


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "./data/fsdd/min_max_values.pkl"


def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    print(sampled_indexes)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print(file_paths)
    print(sampled_min_max_values)
    
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # Sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)

    # Generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)