import torch
import os

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.train_vq import *
from generate import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Optional: faster matmul on Ampere+ GPUs
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))

# Variables
SPECTROGRAMS_SAVE_DIR = "./data/processed/maestro_spectrograms"
SPECTROGRAMS_PATH = "./data/processed/maestro_spectrograms"
MODEL_PATH = "./model/vq_vae_maestro2011/model.pth"

LEARNING_RATE = 1e-5
BATCH_SIZE = 16  # this may need to be small due to memory constraints
EPOCHS = 1000


def load_maestro(path):
    x_train = []
    file_paths = []
    print("Loading spectrograms from:", path)
    for root, _, file_names in os.walk(path):
        print(root)
        for file_name in file_names:
            if file_name.endswith(".npy"):
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
                x_train.append(spectrogram)
                file_paths.append(file_path)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)

    return x_train, file_paths

x_train, _ = load_maestro(SPECTROGRAMS_PATH)
print(x_train.shape)

data_variance = np.var(x_train / 255.0)

# Define VQ-VAE model
VQVAE = VQ_VAE(
    input_shape=(256, x_train.shape[2], 1),
    conv_filters=(256, 128, 64, 32),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(2, 2, 2, (2, 1)),
    #data_variance=data_variance,
    embeddings_size=256,    # K
    latent_space_dim=256    # D
)

# run_options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
# Train the model using the train_model function (with AMP to save memory)
train_model(
    VQVAE,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    data_variance=data_variance,
    save_path=MODEL_PATH,
    amp=True,
    grad_accum_steps=1,
    max_grad_norm=None,
)