import encodings
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Conv2DTranspose, Reshape, Activation, Lambda, Embedding
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

import numpy as np
import pickle
import os 

from vector_quantizer import VectorQuantizer


class VQ_VAE(Model):

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 data_variance,
                 embeddings_size=128,
                 beta=0.25) -> None:
        
        super(VQ_VAE, self).__init__()

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        
        self.num_conv_layers = len(conv_filters)
        self.latent_space_dim = latent_space_dim
        self.shape_before_bottleneck = None
        self.model_input = None
        
        self.encoder = None
        self.vq = None
        self.decoder = None
        self.model = None
        
        # VQ-VAE specifics
        self.data_variance = data_variance

        # From paper:
        
        """We define a latent embedding space e ∈RK×D where K is the size of the discrete latent space (i.e.,
        a K-way categorical), and D is the dimensionality of each latent embedding vector ei. Note that
        there are K embedding vectors ei ∈RD, i ∈1,2,...,K"""
        self._embedding_size = embeddings_size  # K
        self._embedding_dim = latent_space_dim  # D

        """We found the resulting algorithm to be quite robust to β, as 
        the results did not vary for values of β ranging from 0.1 to 2.0.
        We use β = 0.25 in all our experiments, although in general this
        would depend on the scale of reconstruction loss. Since we assume
        a uniform prior for z, the KL term that usually appears in the ELBO
        is constant w.r.t. the encoder parameters and can thus be ignored
        for training."""

        # VQ-VAE commitment parameter
        self._beta = beta
        
        self.set_loss_tracker()
        self.__build_encoder()
        self.__build_quant_layer()
        self.__build_decoder()
        self.__build_vq_vae()


    def summary(self):
        """
        Summarizes the encoder and decoder models using Keras' built-in method.
        """
        self.encoder.summary()
        # self.vq.summary()
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
        
        self.optimizer = optimizer
        super(VQ_VAE, self).compile(self.optimizer)
    

    def call(self, inputs):
        """
        Defines the forward pass of the model. Encodes the input, performs
        vector quantization, and decodes the quantized latent vectors back into
        reconstructions.
        """
        encoder_outputs = self.encoder(inputs)
        quantized_latents = self.vq(encoder_outputs)
        reconstructions = self.decoder(quantized_latents)

        return reconstructions
    

    def train(self, x_train, batch_size, num_epochs):
        """
        Trains the VQ-VAE model on the given data.
        """
        self.fit(x_train,
                 batch_size=batch_size, 
                 epochs=num_epochs,
                 shuffle=True)
    

    def set_loss_tracker(self):
        """
        Initializes loss tracking metrics to monitor during training.
        Tracks total loss, reconstruction loss, and VQ loss.
        """
        self.loss_tracker = {
            "total_loss": tf.keras.metrics.Mean(name="total_loss"),
            "reconstruction_loss": tf.keras.metrics.Mean(name="reconstruction_loss"),
            "vq_loss": tf.keras.metrics.Mean(name="vq_loss"),
        }


    def train_step(self, x):
        """
        Defines the logic for each training step. Computes losses, applies gradients,
        and updates the loss metrics.
        """
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.data_variance
            )
            total_loss = reconstruction_loss + sum(self.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking.
        self.loss_tracker["total_loss"].update_state(total_loss)
        self.loss_tracker["reconstruction_loss"].update_state(reconstruction_loss)
        self.loss_tracker["vq_loss"].update_state(sum(self.losses))

        # Log results.
        return {
            "total_loss": self.loss_tracker["total_loss"].result(),
            "reconstruction_loss": self.loss_tracker["reconstruction_loss"].result(),
            "vq_loss": self.loss_tracker["vq_loss"].result(),
        }
    
    
    def reconstruct(self, input):
        """
        Reconstructs the input data by passing it through the encoder, 
        quantizer, and decoder sequentially.
        """
        encoder_outputs = self.encoder.predict(input)
        quantized_latents = self.vq(encoder_outputs)
        reconstructed = self.decoder.predict(quantized_latents)
        
        return reconstructed, quantized_latents
    

    def save(self, folder="model"):
        """
        Saves the model architecture and weights to the specified folder.
        """
        try:
            self.__create_folder(folder)
            self.__save_parameters(folder)
            self.__save_weights(folder)
            print(f"Model saved successfully in folder: {folder}")
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")


    @classmethod
    def load(cls, save_folder="."):
        """
        Loads the model architecture and weights from the specified folder.
        """
        try:
            # Construct paths for the parameters and weights
            parameters_path = os.path.join(save_folder, "parameters.pkl")
            weights_path = os.path.join(save_folder, ".weights.h5")

            if not os.path.exists(parameters_path):
                raise FileNotFoundError(f"Parameters file not found at: {parameters_path}")
            
            with open(parameters_path, "rb") as f:
                parameters = pickle.load(f)

            variational_autoencoder = VQ_VAE(*parameters)

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
    
   
    
    # <------------------------Private Methods------------------------->

    # <------------------ Encoder ------------------>
    
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
        encoder_outputs = Conv2D(self.latent_space_dim, 1, padding="same")(x)

        return encoder_outputs
    

    # <------------------ Quantization Layer ------------------>
    def __build_quant_layer(self):
        self.vq = VectorQuantizer(self._embedding_size, self._embedding_dim, self._beta)
        

    # <------------------ Decoder ------------------>
    def __build_decoder(self):
        """
        Builds the decoder model, which reconstructs the input from the latent space.
        """
        decoder_input = self.__add_decoder_input()
        # dense_layer = self.__add_dense_layer(decoder_input)
        conv_transpose_layers = self.__add_conv_transpose_layers(decoder_input)
        decoder_output = self.__add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")


    def __add_decoder_input(self):
        """
        Defines the input layer for the decoder, which is the latent space.
        """
        input_layer = Input(shape = self.encoder.output.shape[1:], name = "decoder_input")

        return input_layer
    

    def __add_dense_layer(self, x):
        """
        Adds a dense layer to connect the latent space to the reshaped featured maps.
        """
        num_neurons = np.prod(self.shape_before_bottleneck) # [x, y, z] -> x*y*z
        dense_layer = Dense(num_neurons, name="decoder_dense")(x)
        
        return dense_layer


    def __add_conv_transpose_layers(self, x):
        """
        Adds all transpose convolution layers (up-sampling) in reverse order.
        Excludes the first layer for experimental purposes.
        """
        # Not implementing te first layer (test later)
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


    # <------------------ VQ-VAE Model ------------------>
    def __build_vq_vae(self):
        input = self.model_input
        encoder_output = self.encoder(input)
        quantized_output = self.vq(encoder_output)
        decoder_output = self.decoder(quantized_output)
        
        self.model = Model(input, decoder_output, name = "variational_autoencoder")


    # <------------------ Save/load model ------------------>

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