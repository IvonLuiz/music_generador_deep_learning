from tensorflow.keras.datasets import mnist

import numpy as np
import os
from autoencoder import Autoencoder
from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def load_fsdd(path):
    x_train = []
    
    for root, _, file_names in os.walk():
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]

    return x_train



def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)

    return vae 


if __name__ == "__main__":
    x_train, _, _, _ = load_mnist()
    VAE = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    VAE.save("model")
