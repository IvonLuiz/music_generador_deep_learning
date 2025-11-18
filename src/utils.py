
import os
import numpy as np


def load_maestro(path, target_time_frames=256):
    x_train = []
    file_paths = []
    print("Loading spectrograms from:", path)
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if not file_name.endswith(".npy"):
                pass

            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames)
            
            # Debug: print shapes to see what we're getting
            if len(x_train) < 3:  # Print first few
                print(f"Spectrogram {file_name} shape: {spectrogram.shape}")
            
            # Ensure consistent time dimension - crop or pad to exactly target time frames
            if spectrogram.shape[1] > target_time_frames:
                spectrogram = spectrogram[:, :target_time_frames]  # Crop to 256
            elif spectrogram.shape[1] < target_time_frames:
                # Pad to target time frames
                pad_width = target_time_frames - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            
            x_train.append(spectrogram)
            file_paths.append(file_path)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (N, target_frames, target_frames, 1) (N, 256, 256, 1)
    
    print(f"Final dataset shape: {x_train.shape}")
    return x_train, file_paths


# Helper to locate the min/max entry for a spectrogram file
def find_min_max_for_path(fp, min_max_values, spectrograms_dir):
    bas = os.path.basename(fp)
    candidates = [
        fp,
        os.path.normpath(fp),
        os.path.abspath(fp),
        bas,
        os.path.join(spectrograms_dir, bas),
        os.path.abspath(os.path.join(spectrograms_dir, bas)),
    ]
    # also try with/without leading ./ or ../
    candidates += [c.replace('./', '') for c in list(candidates)]
    candidates += [c.replace('../', '') for c in list(candidates)]

    for c in candidates:
        if c in min_max_values:
            return min_max_values[c]
    # try matching by basename contained in any key
    for k, v in min_max_values.items():
        if bas == os.path.basename(k) or bas in k or os.path.basename(k) in bas:
            return v
    # not found
    return None
