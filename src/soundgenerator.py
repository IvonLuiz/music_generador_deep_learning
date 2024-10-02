from processing.preprocess import MinMaxNormalizer
import os
import pickle
import numpy as np
from vae import VAE
from train import SPECTROGRAMS_PATH, load_fsdd
import librosa

HOP_LENGTH = 256
MIN_MAX_VALUES_PATH = "./data/fsdd/min_max_values.pkl"

class SoundGenerator:

    def __init__(self, autoencoder, hop_length):
        self.autoencoder = autoencoder 
        self.hop_length = hop_length
        self.__min_max_normalizer = MinMaxNormalizer(0, 1)

    
    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.autoencoder.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)

        return signals, latent_representations
    

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
            
        for idx, spectrogram in enumerate(spectrograms):
            spectrogram_path = list(min_max_values.keys())[idx]
            min_max_value = min_max_values.get(spectrogram_path)

            # Debugging
            if min_max_value is None or "min" not in min_max_value or "max" not in min_max_value:
                raise ValueError("MinMax values are missing or malformed")
            
            # Reshape the log spectrogram to remove the third dimension (channels) used for the autoencoder
            log_spectrogram = spectrogram[:, :, 0]
            print(f"Processing: {spectrogram_path}")

            # Apply denormalization
            min_val = min_max_value["min"]
            max_val = min_max_value["max"]
            denorm_log_spec = self.__min_max_normalizer.denormalize(log_spectrogram,  min_val, max_val)
            
            audio_signal = self.__invert_log_spectrogram_to_audio(denorm_log_spec)
            signals.append(audio_signal)
        
        return signals
    

    

    def __invert_log_spectrogram_to_audio(self, log_spectrogram):
        # Log spectrogram (dB) to amplitude domain
        amplitude_spectrogram = librosa.db_to_amplitude(log_spectrogram)
        # STFT to get audio signal
        audio_signal = librosa.istft(amplitude_spectrogram, hop_length=self.hop_length)
        audio_signal = librosa.griffinlim(amplitude_spectrogram, hop_length=self.hop_length)
        
        return audio_signal


if __name__ == "__main__":
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    spectrograms, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    sound_generator.convert_spectrograms_to_audio(spectrograms, min_max_values)