from tensorflow.keras.datasets import mnist

from autoencoder import Autoencoder
from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    # vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    return vae 


if __name__ == "__main__":
    x_train, _, _, _ = load_mnist()
    VAE = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    VAE.save("model")
