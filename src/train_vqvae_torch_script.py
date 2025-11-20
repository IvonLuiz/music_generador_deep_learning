from datetime import datetime
import torch
import os

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.train_vq import *
from generate import *
from utils import load_maestro
from processing.preprocess_audio import TARGET_TIME_FRAMES

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
SPECTROGRAMS_PATH = "./data/processed/maestro_spectrograms_test/"
MODEL_PATH = "./models/vq_vae_maestro2011/"

LEARNING_RATE = 1e-5
BATCH_SIZE = 16  # this may need to be small due to memory constraints
EPOCHS = 1000
current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

x_train, _ = load_maestro(SPECTROGRAMS_PATH, TARGET_TIME_FRAMES, debug_print=False)
print("Input shape: ", x_train.shape)
print("Data range:", x_train.min(), "to", x_train.max())
print("Data samples length:", x_train.shape[0])

# Fix normalization - your preprocessing already normalizes to [0,1]
data_variance = np.var(x_train) # data cines normalized from load_maestro
print(f"Data variance: {data_variance}")

# Stability fix: If variance is too small, the loss term (MSE / 2*var) becomes huge, causing exploding gradients.
# We clamp the variance to a safe minimum (e.g. 0.05) or use 1.0 to rely on standard MSE.
effective_variance = max(float(data_variance), 0.05)
print(f"Effective variance used for training: {effective_variance}")

# Get the actual time dimension from your data
time_frames = x_train.shape[2]
print(f"Time frames detected: {time_frames}")

# Define VQ-VAE model with actual dimensions
# For 257 time frames, we need strides that work well with this dimension
VQVAE = VQ_VAE(
    input_shape=(256, time_frames, 1),
    conv_filters=(256, 128, 64, 32),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=((2, 2), (2, 2), (2, 2), (2, 1)),
    embeddings_size=256,    # K
    latent_space_dim=256    # D
)

# Train the model using the train_model function (with AMP to save memory)
model_path = f"{MODEL_PATH}_{formatted_time}/model.pth"
train_model(
    VQVAE,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    data_variance=effective_variance,
    save_path=MODEL_PATH,
    amp=True,
    grad_accum_steps=1,
    max_grad_norm=1.0, # Add gradient clipping to prevent explosion
)
print("Model training complete. Model saved to:", MODEL_PATH)