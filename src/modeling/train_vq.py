from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

import numpy as np
import os
from autoencoder import Autoencoder
from vae import VAE
from vq_vae import VQ_VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 30

SPECTROGRAMS_PATH = "./data/fsdd/spectrograms/"


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    variance = np.var(x_train / 255.0)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test, variance


def load_fsdd(path):
    x_train = []
    file_paths = []

    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
                x_train.append(spectrogram)
                file_paths.append(file_path)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)

    return x_train, file_paths



def train(x_train, learning_rate, batch_size, epochs, data_variance):
    vq_vae = VQ_VAE(
        input_shape=(28, 28, 1),
        conv_filters=(16, 4),
        conv_kernels=(3, 3),
        conv_strides=(1, 1),
        latent_space_dim=2,
        data_variance=data_variance
    )

    vq_vae.summary()
    vq_vae.compile(learning_rate)
    print(x_train.shape)
    print(type(x_train))
    # vq_vae.set_loss_tracker(data_variance)
    vq_vae.train(x_train, batch_size, epochs)

    return vq_vae 


if __name__ == "__main__":
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)



    vq_vae = VQ_VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64),
        conv_kernels=(3, 3),
        conv_strides=(1, 1),
        latent_space_dim=16,
        data_variance=data_variance,
        embeddings_size=128
    )
    vq_vae.summary()
    # vq_vae = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vq_vae.compile(optimizer=Adam())
    vq_vae.fit(x_train_scaled, epochs=30, batch_size=128)
    # x_train, y_train, x_test, y_test, variance = load_mnist()
    # vq_vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS, data_variance)
    # vq_vae.save("model")
