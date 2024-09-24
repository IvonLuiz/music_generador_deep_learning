from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Lambda
class Encoder():

    def __init__(self, input_shape) -> None:

        self.inputs = self.input_shape
        self.conv = []
        
        self.build_encoder()


    def build_encoder(self):
        encoder_input = self.set_encoder_input(self, self.input_shape)
        conv_layers = self.set_conv_layers(self.num_conv_layers, encoder_input)
        self.encoder = Model(conv_layers, name = "encoder")


    def set_encoder_input(self, shape):
        self.input = Input(shape, name="encoder_input")
        

    def set_conv_layers(self, num_layers, x):
        for index in range(num_layers):
           x = self.add_conv_layer(self.filters[index], index)(x)


    def add_conv_layer(self, filters, index):

        conv_layer = Conv2D(
            filters = self.conv_filters[index],
            kernel_size = self.conv_kernels[index],
            strides = self.conv_strides[index],
            padding = "same",
            name = f"conv_layer_{index}"
        )
        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu_{index}")(x)
        x = BatchNormalization(name=f"encoder_bn_f{index}")
    
    
