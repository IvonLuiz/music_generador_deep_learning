from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Conv2DTranspose, Reshape, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

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
        # From paper:
        """We found the resulting algorithm to be quite robust to β, as 
        the results did not vary for values of β ranging from 0.1 to 2.0.
        We use β = 0.25 in all our experiments, although in general this
        would depend on the scale of reconstruction loss. Since we assume
        a uniform prior for z, the KL term that usually appears in the ELBO
        is constant w.r.t. the encoder parameters and can thus be ignored
        for training."""

        self.codebook = None
        self.beta = 0.2
        
        self.__build_codebook()
        self.__build_encoder()
        self.__build_decoder()
        self.__build_variational_autoencoder()
    
    # def forward(self, x):
    #     encoder_output = self.encoder(x)
        


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
    

    def __build_codebook(self, K, M):

        self.codebook = []
        rand


    def __add_bottleneck(self, x):
        """
        Output of the encoder. Defines the bottleneck layer by flattening the data
        and adding a bottleneck with Gaussian sampling dense layer.
        """
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        # x = Flatten()(x)




        fnp.argmin()
        # self.mu = Dense(self.latent_space_dim, name="z_mu")(x)
        # self.log_var = Dense(self.latent_space_dim, name="z_log_variance")(x)

        # z = Sampling()([self.mu, self.log_var])

        # z = Lambda(Sampling, name="encoder_output",
        #            output_shape=(self.latent_space_dim,))([self.mu, self.log_var])

        return z


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

