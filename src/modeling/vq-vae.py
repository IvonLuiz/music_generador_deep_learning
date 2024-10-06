import encodings
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Conv2DTranspose, Reshape, Activation, Lambda, Embedding, VectorQuantizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import tensorflow as tf

import numpy as np


class VQ_VAE():

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim) -> None:
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        
        self.num_conv_layers = len(conv_filters)
        self.latent_space_dim = latent_space_dim
        self.shape_before_bottleneck = None
        self.model_input = None
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        # VQ-VAE specifics
        self.pre_quant_conv_layer = None
        self.post_quant_conv_layer = None
        self.vq = None

        # From paper:
        
        """We define a latent embedding space e ∈RK×D where K is the size of the discrete latent space (i.e.,
        a K-way categorical), and D is the dimensionality of each latent embedding vector ei. Note that
        there are K embedding vectors ei ∈RD, i ∈1,2,...,K"""
        self.__embedding_size = 3 # K
        self.__embedding_dim = 2 # D

        self.codebook = None

        """We found the resulting algorithm to be quite robust to β, as 
        the results did not vary for values of β ranging from 0.1 to 2.0.
        We use β = 0.25 in all our experiments, although in general this
        would depend on the scale of reconstruction loss. Since we assume
        a uniform prior for z, the KL term that usually appears in the ELBO
        is constant w.r.t. the encoder parameters and can thus be ignored
        for training."""
        self.beta = 0.2
        
        # self.__build_codebook()
        self.__build_encoder()
        self.__build_quant_layer()
        self.__build_decoder()
        self.__build_vq_vae()
    

    def forward(self, x):
        input = self.model_input
        encoder_output = self.encoder(input)
        quant_input = self.pre_quant_conv_layer(encoder_output)

        # Quantization
        B, C, H, W = quant_input.shape
        decoder_output = self.decoder(encoder)
        self.model = Model(input, decoder_output, name = "variational_autoencoder")




    # <------------------------Private Methods------------------------->

    # ENCODER:
    
    def __build_encoder(self):
        """
        Builds the encoder model, which maps the input to the latent space.
        Uses the arguments passed when instantiating the class to create the input
        layer and the convolutional layers. The bottleneck will be the output.        
        """
        encoder_input = self.__add_encoder_input(self.input_shape)
        conv_layers = self.__add_conv_layers(self.num_conv_layers, encoder_input)
        bottleneck = self.__add_bottleneck(self, conv_layers)

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
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Conv2D(self.latent_space_dim, 1, padding="same")(x)

        return x
    

    # CODEBOOK
    def __build_quant_layer(self):
        """
        Output of the encoder. Defines the bottleneck layer by flattening the data
        and adding a bottleneck with Gaussian sampling dense layer.
        """
        self.pre_quant_conv_layer = Conv2D(
            filters = self.conv_filters[-1],
            kernel_size = 1,
            strides = 1,
            padding = "same",
            name = f"pre_quant_conv_layer"
        )
        self.vq = VectorQuantizer(self.__embedding_dim, self.__embedding_size)

        # x = pre_quant_conv_layer(x)

        # self.embedding = Embedding(input_dim=self.__embedding_size,
        #                            output_dim=self.__embedding_dim,
        #                            name="embedding_layer")
        # x = self.embedding(x)

        # self.post_quant_conv_layer = Conv2D(
        #     filters = self.conv_filters[-1],
        #     kernel_size = self.conv_kernels[-1],
        #     # strides = self.conv_strides[index],
        #     padding = "same",
        #     name = f"post_quant_conv_layer"
        # )
        # x = post_quant_conv_layer(x)
        

    # DECODER:

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


    # Full model
    def __build_vq_vae(self):
        input = self.model_input
        encoder = self.encoder(input)
        vq = self.vq(encoder)
        decoder_output = self.decoder(vq)
        
        self.model = Model(input, decoder_output, name = "variational_autoencoder")


class VectorQuantizer(tf.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim     # D
        self._num_embeddings = num_embeddings   # K
        
        # Initialize embeddings with inform random values in interval (-1/K, 1/K)
        initializer = tf.random_uniform_initializer(minval=-1/self._num_embeddings,
                                                    maxval=1/self._num_embeddings)
        self._embedding = tf.Variable(
            initializer(shape=(self._num_embeddings, self._embedding_dim)),
            trainable=True,
            name="embedding_vectors"
        )
        # self._embedding = Embedding(self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = tf.reshape(inputs, [-1, self._embedding_dim])
        
        # Calculate distances
        distances = (
            tf.reduce_sum(flat_input**2, axis=1, keepdims=True)
            + tf.reduce_sum(tf.pow(self._embedding.weights))
            - 2 * tf.matmul(flat_input, self._embedding.weights)
        )

        # Encoding
        encoding_indices = tf.expand_dims(tf.argmin(distances, axis=1), axis=1)
        encodings = tf.zeros((tf.shape(encoding_indices)[0], self._num_embeddings))
        encodings = tf.one_hot(tf.squeeze(encoding_indices, axis=1), depth=self._num_embeddings)

        # Quantize and unflatten
        quantized = tf.matmul(encodings, self._embedding.weights)
        quantized = tf.reshape(quantized, input_shape)

        # Loss
        # Maybe use tf.reduce_sum idk
        e_latent_loss = MSE(tf.stop_gradient(quantized), inputs)
        q_latent_loss = MSE(quantized, tf.stop_gradient(inputs))
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        
        # Convert quantized back from BHWC to BCHW
        quantized = tf.transpose(quantized, perm=[0, 3, 1, 2])

        return loss, quantized, perplexity, encodings