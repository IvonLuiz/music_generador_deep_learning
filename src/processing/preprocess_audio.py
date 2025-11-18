import os 
import pickle
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class Loader:
    """
    Loader is responsible for loading an audio file.
    """

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class Padder:
    """
    Padder is responsible to apply padding to an array.
    """

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length


    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormalizer:
    """MinMaxnormalizer applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class SpectrogramVisualizer:
    """
    SpectrogramVisualizer saves spectrograms as image files for visual inspection.
    """
    
    def __init__(self, visualization_save_dir):
        self.visualization_save_dir = visualization_save_dir
        Path(self.visualization_save_dir).mkdir(parents=True, exist_ok=True)
    
    def save_spectrogram_image(self, spectrogram, file_path, prefix="", dpi=150):
        """
        Save spectrogram as PNG image.
        
        Args:
            spectrogram: 2D numpy array (frequency_bins, time_frames)
            file_path: Original audio file path (used to generate image name)
            prefix: Optional prefix for the image filename
            dpi: Image resolution (dots per inch)
        """
        # Generate save path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        image_name = f"{prefix}{file_name}_spectrogram.png"
        save_path = os.path.join(self.visualization_save_dir, image_name)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.imshow(
            spectrogram, 
            aspect='auto', 
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
        plt.title(f'Log Spectrogram: {file_name}')
        plt.tight_layout()
        
        # Save the image
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory
        
        return save_path
    
    def save_comparison_image(self, original_spec, normalized_spec, file_path, dpi=150):
        """
        Save side-by-side comparison of original and normalized spectrograms.
        
        Args:
            original_spec: Original log spectrogram
            normalized_spec: Normalized spectrogram [0,1]
            file_path: Original audio file path
            dpi: Image resolution
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        image_name = f"{file_name}_comparison.png"
        save_path = os.path.join(self.visualization_save_dir, image_name)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original spectrogram
        im1 = ax1.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f'Original Log Spectrogram\n{file_name}')
        ax1.set_xlabel('Time Frames')
        ax1.set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
        
        # Normalized spectrogram
        im2 = ax2.imshow(normalized_spec, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax2.set_title(f'Normalized Spectrogram [0,1]\n{file_name}')
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=ax2, label='Normalized Magnitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        Path(self.feature_save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.min_max_values_save_dir).mkdir(parents=True, exist_ok=True)


    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)


    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalize spectrogram
        5- save the normalized spectrogram
        6- optionally save visualization images

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self, enable_visualization=False):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.visualizer = None
        self.enable_visualization = enable_visualization
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)
                    print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
        
        # Optional visualization
        if self.enable_visualization and self.visualizer is not None:
            # Save comparison image (original vs normalized)
            self.visualizer.save_comparison_image(feature, norm_feature, file_path)
            print(f"Saved visualization for {os.path.basename(file_path)}")

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    SAMPLE_RATE = 22050  # Match the sample rate expected by SoundGenerator
    # Calculate duration for more meaningful music segments
    # For 128 time frames: (128 * 256) / 22050 ≈ 1.49 seconds
    # For 256 time frames: (256 * 256) / 22050 ≈ 2.97 seconds  
    # For 512 time frames: (512 * 256) / 22050 ≈ 5.95 seconds
    
    TARGET_TIME_FRAMES = 256  # Adjust this for desired length
    DURATION = (TARGET_TIME_FRAMES * 256) / SAMPLE_RATE # ~2.97 seconds for 256 frames
    print(f"Using {TARGET_TIME_FRAMES} time frames = {DURATION:.2f} seconds")
    
    MONO = True

    # SPECTROGRAMS_SAVE_DIR = "./data/fsdd/spectrograms/"
    # MIN_MAX_VALUES_SAVE_DIR = "./data/fsdd/"
    # FILES_DIR = "./data/fsdd/audio/"
    
    FILES_DIR = "./data/raw/maestro-v3.0.0/2011"
    SPECTROGRAMS_SAVE_DIR = "./data/processed/maestro_spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = "./data/raw/maestro-v3.0.0/2011/"
    VISUALIZATION_SAVE_DIR = "./data/visualizations/spectrograms/"
    
    # Enable visualization (set to False to disable)
    ENABLE_VISUALIZATION = True
    
    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    # Normalize to [0, 1] range as expected by VQ-VAE
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    
    # Create visualizer if enabled
    visualizer = None
    if ENABLE_VISUALIZATION:
        visualizer = SpectrogramVisualizer(VISUALIZATION_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline(enable_visualization=ENABLE_VISUALIZATION)
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.visualizer = visualizer

    preprocessing_pipeline.process(FILES_DIR)