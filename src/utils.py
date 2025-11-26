import os
import numpy as np
from tqdm import tqdm
import yaml


def load_maestro(path, target_time_frames=256, debug_print=False):
    # Check for cached dataset to speed up loading
    # We use the parent directory of the path if path is a directory, or just path
    cache_dir = path if os.path.isdir(path) else os.path.dirname(path)
    cache_path = os.path.join(cache_dir, "dataset_cache.npz")
    
    if os.path.exists(cache_path):
        print(f"Found cached dataset at {cache_path}. Loading...")
        try:
            data = np.load(cache_path)
            x_train = data['x_train']
            file_paths = data['file_paths']
            print(f"Loaded cached dataset shape: {x_train.shape}")
            return x_train, file_paths
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from source files.")

    x_train = []
    file_paths = []
    
    # First, collect all file paths to know the total for tqdm
    all_files = []
    print("Scanning files in:", path)
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                all_files.append(os.path.join(root, file_name))

    print(f"Found {len(all_files)} spectrograms. Loading...")

    for file_path in tqdm(all_files, desc="Loading spectrograms"):
        try:
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
        except ValueError as ve:
            print(f"ValueError loading {file_path}: {ve}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        # Debug: print shapes to see what we're getting
        if debug_print:
            print(f"Spectrogram {os.path.basename(file_path)} shape: {spectrogram.shape}")
            
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
    
    # Save to cache for next time
    print(f"Saving dataset to cache {cache_path}...")
    np.savez(cache_path, x_train=x_train, file_paths=file_paths)
    print("Cache saved.")
    
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

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
