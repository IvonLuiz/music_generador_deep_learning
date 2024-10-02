import numpy as np






def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print(file_paths)
    print(sampled_min_max_values)
    
    return sampled_spectrogrmas, sampled_min_max_values
