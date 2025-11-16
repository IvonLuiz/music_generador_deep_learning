import librosa

from processing.preprocess_audio import MinMaxNormalizer

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

            for spectrogram, min_max_value in zip(spectrograms, min_max_values):
                # Reshape the log spectrogram to remove the third dimension (channels) used for the autoencoder
                log_spectrogram = spectrogram[:, :, 0]
                
                denorm_log_spec = self.__min_max_normalizer.denormalize(
                    log_spectrogram, min_max_value["min"], min_max_value["max"]
                )

                signal = self.__invert_log_spectrogram_to_audio(denorm_log_spec)
                signals.append(signal)

            return signals
    

    def __invert_log_spectrogram_to_audio(self, log_spectrogram):
        # Log spectrogram (dB) to amplitude domain
        amplitude_spectrogram = librosa.db_to_amplitude(log_spectrogram)
        # STFT to get audio signal
        audio_signal = librosa.istft(amplitude_spectrogram, hop_length=self.hop_length)
        # audio_signal = librosa.griffinlim(amplitude_spectrogram, hop_length=self.hop_length)
        
        return audio_signal
