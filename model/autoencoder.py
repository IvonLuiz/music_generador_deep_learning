from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Lambda, Conv2DTranspose
from tensorflow.keras import backend as K
import numpy as np

class Encoder():

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        
        self.num_conv_layers = len(conv_filters)
        self.latent_space_dim = latent_space_dim
        self.shape_before_bottleneck = None

        self.build_encoder()


    def summary(self):
        """
        Summary the model with keras method
        """
        self.encoder.summary()
    

    def build_encoder(self):
        """
        Builds the encoder with the keras model.
        Uses the arguments passed when instantiating the class to create the input
        layer and the convolutional layers. The bottleneck will be the output.        
        """
        encoder_input = self.set_encoder_input(self.input_shape)
        conv_layers = self.set_conv_layers(self.num_conv_layers, encoder_input)
        bottleneck = self.set_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")


    def set_encoder_input(self, shape):
        """
        Sets the encoder input using the Input method by Keras.
        """
        input = self.input = Input(shape, name="encoder_input")
        return input


    def set_conv_layers(self, num_layers, x):
        """
        Loops over the number of desired layers and adds convolutional blocks.
        """

        for index in range(num_layers):
           x = self.add_conv_layer(index, x)
        
        return x


    def add_conv_layer(self, index, x):
        """
        Adds a convolutional block to a graph of layers. It has 3 parts:
        A convolutional kernel over a 2D spatial dimension (Conv 2D);
        Rectified linear activation unit (ReLU);
        Batch normalization.
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
        x = BatchNormalization(name=f"encoder_bn_f{index}")(x)

        return x
    

    def set_bottleneck(self, x):
        """
        Output of the encoder. Flattens the data and add bottleneck (Dense layer).
        """

        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)

        return x


    def build_decoder(self, x):

        decoder_input = self.set_decoder_input()
        dense_layer = self.add_dense_layer(decoder_input)
        reshaped_layer = self.add_reshape_layer(dense_layer)
        conv_transpose_layers = self.add_conv_transpose_layers(reshaped_layer)
        decoder_output = self.add_output_layer(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")


    def add_decoder_input(self):
        input_layer = Input(shape = self.latent_space_dim, name = "decoder_input")
        return input_layer
    

    def add_dense_layer(self, x):
        
        num_neurons = np.prod(self.shape_before_bottleneck) # [x, y, z] -> x*y*z
        dense_layer = Dense(num_neurons, name="decoder_dense")(x)
        
        return dense_layer


    def set_dense_layers(self, x):
        for index in range(self.num_conv_layers):
           x = self.add_dense_layer(index, x)
        
        return x

    
    def add_dense_layer(self, index, x):

        conv_layer = Conv2DTranspose(
            filters = self.conv_filters[index],
            kernel_size = self.conv_kernels[index],
            strides = self.conv_strides[index],
            padding = "same",
            name = f"conv_layer_{index}"
        )
        x = conv_layer(x)
        x = ReLU(name = f"decoder_relu_{index}")(x)
        x = BatchNormalization(name=f"decoder_bn_f{index}")(x)


        


if __name__ == "__main__":
    autoencoder = Encoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()