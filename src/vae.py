from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Conv2DTranspose, Reshape, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer

import tensorflow as tf
import keras
from keras import ops
from keras import layers

import numpy as np
import os
import pickle


def square_fn(x):
    return tf.square(x)

def exp_fn(x):
    return tf.exp(x)

def reduce_sum_fn(x):
    return tf.reduce_sum(x, axis=1)

def kl_loss(mu, log_var):
    return -0.5 * reduce_sum_fn(1 + log_var - square_fn(mu) - exp_fn(log_var))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), 
                                  mean=0.,
                                  stddev=1.,
                                  seed=self.seed_generator)
        
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VAE():
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 **kwargs):
        """
        Initializes the variational autoencoder with the provided parameters.
        
        Arguments:
        - input_shape: Shape of the input image.
        - conv_filters: List of filters for each convolutional layer.
        - conv_kernels: List of kernel sizes for each convolutional layer.
        - conv_strides: List of strides for each convolutional layer.
        - latent_space_dim: Size of the latent space (bottleneck).
        """
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        
        self.num_conv_layers = len(conv_filters)
        self.latent_space_dim = latent_space_dim
        self.w_rec_loss = 1000000
        self.shape_before_bottleneck = None
        self.model_input = None
        
        self.encoder = None
        self.decoder = None
        self.model = None

        self.__build_encoder()
        self.__build_decoder()
        self.__build_variational_autoencoder()


    def summary(self):
        """
        Summarizes the encoder and decoder models using Keras' built-in method.
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def compile(self, learning_rate=0.0001, optimizer=None):
        """
        Compiles the autoencoder model by setting the optimizer and loss function.
        
        Args:
        - learning_rate: Learning rate for the optimizer.
        - optimizer: Optimizer to use (default is Adam).
        """
        if optimizer is None:
            optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss=self.__calculate_combined_loss,
                           metrics=[self.__calculate_reconstruction_loss,
                                    self.__calculate_kl_loss])


    def train(self, x_train, batch_size, num_epochs):

        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size, 
                       epochs=num_epochs,
                       shuffle=True)


    def save(self, folder="model"):
        try:
            self.__create_folder(folder)
            self.__save_parameters(folder)
            self.__save_weights(folder)
            print(f"Model saved successfully in folder: {folder}")
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")


    @classmethod
    def load(cls, save_folder="."):
        try:
            # Construct paths for the parameters and weights
            parameters_path = os.path.join(save_folder, "parameters.pkl")
            weights_path = os.path.join(save_folder, ".weights.h5")

            if not os.path.exists(parameters_path):
                raise FileNotFoundError(f"Parameters file not found at: {parameters_path}")
            
            with open(parameters_path, "rb") as f:
                parameters = pickle.load(f)

            variational_autoencoder = VAE(*parameters)

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at: {weights_path}")

            variational_autoencoder.__load_weights(weights_path)

            return variational_autoencoder

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except (pickle.UnpicklingError, IOError) as e:
            print(f"Error loading parameters or weights: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None
    

    def reconstruct(self, input):
        latent_representations = self.encoder.predict(input)
        reconstructed = self.decoder.predict(latent_representations)
        
        return reconstructed, latent_representations


    # Private methods

    ## Loss calculations
    # def __calculate_reconstruction_loss(self, y_true, y_pred):
    #     reconstruc_loss = ops.mean(
    #         ops.sum(
    #             keras.losses.binary_crossentropy(y_true, y_pred),
    #             axis=(1, 2),
    #         )
    #     )

    #     return reconstruc_loss
    def __calculate_reconstruction_loss(self, y_true, y_pred):
        error = y_true - y_pred
        reconstruction_loss = ops.mean(ops.square(error), axis=[1, 2, 3])
        return reconstruction_loss


    def __calculate_kl_loss(self, y_true, y_pred):
        return Lambda(lambda inputs: kl_loss(inputs[0], inputs[1]), 
                            name="kl_loss_output",
                            output_shape=())([y_true, y_pred])
    

    def __calculate_combined_loss(self, y_true, y_pred):

        reconstructed_loss = self.__calculate_reconstruction_loss(y_true, y_pred)
        kl_loss = self.__calculate_kl_loss(y_true, y_pred)
        combined_loss = self.w_rec_loss * reconstructed_loss + kl_loss

        return combined_loss

    # Methods to save/load model
    def __create_folder(self, folder="model"):
        if not os.path.exists(folder):
            os.makedirs(folder)


    def __save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)


    def __save_weights(self, save_folder):
        save_path = os.path.join(save_folder, ".weights.h5")
        self.model.save_weights(save_path)


    def __load_weights(self, weights_path):
        self.model.load_weights(weights_path)
    

    """--------ENCODER--------"""

    def __build_encoder(self):
        """
        Builds the encoder model, which maps the input to the latent space.
        Uses the arguments passed when instantiating the class to create the input
        layer and the convolutional layers. The bottleneck will be the output.        
        """
        encoder_input = self.__add_encoder_input(self.input_shape)
        conv_layers = self.__add_conv_layers(self.num_conv_layers, encoder_input)
        bottleneck = self.__add_bottleneck(conv_layers)

        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")


    def __add_encoder_input(self, shape):
        """
        Defines the input layer for the encoder using the Input method by Keras.
        """
        input = Input(shape, name="encoder_input")
        return input


    def __add_conv_layers(self, num_layers, x):
        """
        Loops over the number of desired layers and adds convolutional blocks.
        """
        for index in range(num_layers):
           x = self.__add_conv_layer(index, x)
        
        return x


    def __add_conv_layer(self, index, x):
        """
        Adds a convolutional block to a graph of layers. It has 3 parts:
        A convolutional kernel over a 2D spatial dimension (Conv 2D);
        Rectified linear activation unit (ReLU);
        Batch normalization (BatchNorm).
        """
        conv_layer = Conv2D(
            filters = self.conv_filters[index],
            kernel_size = self.conv_kernels[index],
            strides = self.conv_strides[index],
            padding = "same",
            name = f"conv_layer_{index}"
        )
        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu_{index}")(x)
        x = BatchNormalization(name=f"encoder_bn_{index}")(x)

        return x
    

    def __add_bottleneck(self, x):
        """
        Output of the encoder. Defines the bottleneck layer by flattening the data
        and adding a bottleneck with Gaussian sampling dense layer.
        """
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        
        self.mu = Dense(self.latent_space_dim, name="z_mu")(x)
        self.log_var = Dense(self.latent_space_dim, name="z_log_variance")(x)

        z = Sampling()([self.mu, self.log_var])

        # z = Lambda(Sampling, name="encoder_output",
        #            output_shape=(self.latent_space_dim,))([self.mu, self.log_var])

        return z


    """--------DECODER--------"""

    def __build_decoder(self):
        """
        Builds the decoder model, which reconstructs the input from the latent space.
        """
        decoder_input = self.__add_decoder_input()
        dense_layer = self.__add_dense_layer(decoder_input)
        reshaped_layer = self.__add_reshape_layer(dense_layer)
        conv_transpose_layers = self.__add_conv_transpose_layers(reshaped_layer)
        decoder_output = self.__add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")


    def __add_decoder_input(self):
        """
        Defines the input layer for the decoder, which is the latent space.
        """
        input_layer = Input(shape = (self.latent_space_dim,), name = "decoder_input")

        return input_layer
    

    def __add_dense_layer(self, x):
        """
        Adds a dense layer to connect the latent space to the reshaped featured maps.
        """
        num_neurons = np.prod(self.shape_before_bottleneck) # [x, y, z] -> x*y*z
        dense_layer = Dense(num_neurons, name="decoder_dense")(x)
        
        return dense_layer


    def __add_reshape_layer(self, x):
        """
        Reshapes the dense output into the original feature map shape before convolution.
        """
        reshape_layer = Reshape(self.shape_before_bottleneck)(x)
        
        return reshape_layer


    def __add_conv_transpose_layers(self, x):
        """
        Adds all transpose convolution layers (up-sampling) in reverse order.
        Excludes the first layer for experimental purposes.
        """
        # Not implemeting te first layer (test later)
        for index in reversed(range(1, self.num_conv_layers)):
            x = self.__add_conv_transpose_layer(index, x)
        
        return x
    

    def __add_conv_transpose_layer(self, index, x):
        """
        Adds a single Conv2DTranspose block (Conv2DTranspose, ReLU, BatchNorm).
        """
        layer_num = self.num_conv_layers - index

        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[index],
            kernel_size = self.conv_kernels[index],
            strides = self.conv_strides[index],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name = f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)

        return x


    def __add_decoder_output(self, x):
        """
        Adds the final output layer to the decoder with a sigmoid activation.
        """
        layer = Conv2DTranspose(
            filters=1,  # One output channel from spectrograms (grayscale)
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_output_layer_{self.num_conv_layers}"
        )
        x = layer(x)
        x = Activation("sigmoid", name="sigmoid_output")(x)

        return x

    # Full model
    def __build_variational_autoencoder(self):
        input = self.model_input
        encoder = self.encoder(input)
        decoder_output = self.decoder(encoder)
        self.model = Model(input, decoder_output, name = "variational_autoencoder")
