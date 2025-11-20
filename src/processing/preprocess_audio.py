import os 
import pickle
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

TARGET_TIME_FRAMES = 256  # Adjust this for desired length

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
                            hop_length=self.hop_length)[:-1]  # Slice to remove Nyquist bin (257 -> 256) for power-of-2 height
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
        self._padder = None
        self._extractor = None
        self._normalizer = None
        self._saver = None
        self._visualizer = None
        self._loader = None
        self.enable_visualization = enable_visualization
        self.min_max_values = {}
        self._num_expected_samples = 0 

    @property
    def loader(self):
        return self._loader
    
    @property
    def padder(self):
        return self._padder

    @property
    def extractor(self):
        return self._extractor

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def saver(self):
        return self._saver

    @property
    def visualizer(self):
        return self._visualizer

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    @padder.setter
    def padder(self, padder):
        self._padder = padder
    
    @extractor.setter
    def extractor(self, extractor):
        self._extractor = extractor

    @normalizer.setter
    def normalizer(self, normalizer):
        self._normalizer = normalizer

    @saver.setter
    def saver(self, saver):
        self._saver = saver

    @visualizer.setter
    def visualizer(self, visualizer):
        self._visualizer = visualizer

    def process(self, audio_files_dir):
        """Process all audio files in a directory."""
        if self.saver is None:
            print("Error: Saver not set.")
            return

        total_segments = 0
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    segments = self._process_file_with_segments(file_path)
                    total_segments += segments
                    print(f"Processed {file} -> {segments} segments")
        print(f"Total segments created: {total_segments}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file_with_segments(self, file_path):
        """
        Process a single audio file by extracting multiple overlapping segments.
        This maximizes the use of available musical content.
        """
        if self.loader is None:
            print("Error: Loader not set.")
            return 0

        segments_created = 0

        # Load the full audio file without duration limit
        full_signal = librosa.load(file_path, sr=self.loader.sample_rate, mono=self.loader.mono)[0]
            
        # Calculate segment parameters
        segment_samples = self._num_expected_samples
        hop_samples = segment_samples // 2  # 50% overlap between segments
        
        # Extract overlapping segments
        for start_idx in range(0, len(full_signal) - segment_samples + 1, hop_samples):

            # Extract segment
            signal_segment = full_signal[start_idx:start_idx + segment_samples]
            
            # Skip if segment is too short (shouldn't happen with the range, but safety check)
            if len(signal_segment) < segment_samples:
                signal_segment = self._apply_padding(signal_segment)
            
            # Create unique filename for this segment
            segment_file_path = f"{file_path}_segment_{segments_created:03d}"
            # Extract spectrogram features and save
            feature, norm_feature, segment_file_path = self._extract_spectrogram_segment(signal_segment, segment_file_path)
    
            # Optional visualization (only for first few segments to avoid too many images)
            if (self.enable_visualization and self.visualizer is not None and 
                segments_created < 3):  # Only visualize first 3 segments per file
                self.visualizer.save_comparison_image(feature, norm_feature, segment_file_path)
            
            segments_created += 1
        
        # If no segments were created (file too short), process the whole file with padding
        if segments_created == 0:
            segments_created = self._process_single_segment(file_path, full_signal)
        
        return segments_created

    def _extract_spectrogram_segment(self, signal_segment, segment_file_path):
        """Extract spectrogram from a signal segment."""
        if self.extractor is None or self.normalizer is None or self.saver is None:
            print("Error: One or more components not set properly.")
            return None, None, None

        feature = self.extractor.extract(signal_segment)
        norm_feature = self.normalizer.normalize(feature)
        # Save 
        save_path = self.saver.save_feature(norm_feature, segment_file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

        return feature, norm_feature, segment_file_path

    def _process_single_segment(self, file_path, signal):
        """Process a single segment (used for very short audio files)."""
        # Early exit if components not set
        if self.extractor is None or self.normalizer is None or self.saver is None:
            print("Error: One or more components not set properly.")
            return 0
            
        if len(signal) < self._num_expected_samples:
            signal = self._apply_padding(signal)
        elif len(signal) > self._num_expected_samples:
            signal = signal[:self._num_expected_samples]  # Truncate if too long
            
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
        
        # Optional visualization
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.save_comparison_image(feature, norm_feature, file_path)
            print(f"Saved visualization for {os.path.basename(file_path)}")

        return 1

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        if self.padder is None:
            print("Error: Padder not set.")
            return signal

        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    SAMPLE_RATE = 22050    # samples per second
    FRAME_SIZE = 512       # samples for each STFT window
    HOP_LENGTH = 256       # move amount of samples between windows
    # Calculate duration for meaningful music segments
    # For 128 time frames: (128 * 256) / 22050 ≈ 1.49 seconds
    # For 256 time frames: (256 * 256) / 22050 ≈ 2.97 seconds  
    # For 512 time frames: (512 * 256) / 22050 ≈ 5.95 seconds
    
    DURATION = (TARGET_TIME_FRAMES * HOP_LENGTH) / SAMPLE_RATE # ~2.97 seconds for 256 frames
    
    print(f"Using {TARGET_TIME_FRAMES} time frames = {DURATION:.2f} seconds per segment")
    print(f"With 50% overlap, a 3-minute song will generate ~{int(180/DURATION*2)} segments!")
    print(f"Expected spectrogram shape per segment: (256, {TARGET_TIME_FRAMES})")
    
    MONO = True

    # SPECTROGRAMS_SAVE_DIR = "./data/fsdd/spectrograms/"
    # MIN_MAX_VALUES_SAVE_DIR = "./data/fsdd/"
    # FILES_DIR = "./data/fsdd/audio/"
    
    FILES_DIR = "./data/raw/maestro-v3.0.0/2011"
    SPECTROGRAMS_SAVE_DIR = "./data/processed/maestro_spectrograms_test/"
    MIN_MAX_VALUES_SAVE_DIR = "./data/processed/maestro_spectrograms_test/min_max_values/"
    VISUALIZATION_SAVE_DIR = "./data/processed/maestro_spectrograms_test/spectrograms/"
    
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