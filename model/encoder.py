from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Lambda
from tensorflow.keras import backend as K


class Encoder():

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides):
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        
        self.num_conv_layers = len(conv_filters)

        self.build_encoder()

    def summary(self):
        self.encoder.summary()
    

    def build_encoder(self):
        encoder_input = self.set_encoder_input(self.input_shape)
        conv_layers = self.set_conv_layers(self.num_conv_layers, encoder_input)
        self.encoder = Model(encoder_input, conv_layers, name = "encoder")


    def set_encoder_input(self, shape):
        input = self.input = Input(shape, name="encoder_input")
        return input


    def set_conv_layers(self, num_layers, x):
        for index in range(num_layers):
           x = self.add_conv_layer(index, x)
        return x


    def add_conv_layer(self, index, x):

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
    
if __name__ == "__main__":
    autoencoder = Encoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1)
    )
    autoencoder.summary()