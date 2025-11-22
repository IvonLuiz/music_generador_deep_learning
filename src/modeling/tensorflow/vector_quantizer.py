import tensorflow as tf
from tensorflow.keras.losses import MSE


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        
        self._embedding_dim = embedding_dim      # D
        self._num_embeddings = num_embeddings    # K

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self._beta = beta
        
        # Initialize embeddings with inform random values
        initializer = tf.random_uniform_initializer()
        self._embedding = tf.Variable(
            initializer(shape=(self._num_embeddings, self._embedding_dim), dtype="float32"),
            trainable=True,
            name="embedding_vectors"
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        
        # Flatten input keeping `embedding_dim` intact.
        flat_input = tf.reshape(inputs, [-1, self._embedding_dim])
        
        # Calculate distances
        distances = (
            tf.reduce_sum(flat_input ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self._embedding ** 2, axis=1)
            - 2 * tf.matmul(flat_input, self._embedding, transpose_b=True)
        )

        # Encoding
        encoding_indices = tf.expand_dims(tf.argmin(distances, axis=1), axis=1)
        encodings = tf.zeros((tf.shape(encoding_indices)[0], self._num_embeddings))
        encodings = tf.one_hot(tf.squeeze(encoding_indices, axis=1), depth=self._num_embeddings)

        # Quantize and unflatten
        quantized = tf.matmul(encodings, self._embedding)
        quantized = tf.reshape(quantized, input_shape)

        # Loss
        # Maybe use tf.reduce_sum idk
        commitment_loss = MSE(tf.stop_gradient(quantized), inputs)
        codebook_loss = MSE(quantized, tf.stop_gradient(inputs))
        self.add_loss(codebook_loss + self._beta * commitment_loss)
        
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized

