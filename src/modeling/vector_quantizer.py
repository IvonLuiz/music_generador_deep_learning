import tensorflow as tf
from tensorflow.keras.losses import MSE


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        
        self._embedding_dim = embedding_dim     # D
        self._num_embeddings = num_embeddings   # K
        self.loss = None
        
        # Initialize embeddings with inform random values in interval (-1/K, 1/K)
        initializer = tf.random_uniform_initializer(minval=-1/self._num_embeddings,
                                                    maxval=1/self._num_embeddings)
        self._embedding = tf.Variable(
            initializer(shape=(self._num_embeddings, self._embedding_dim)),
            trainable=True,
            name="embedding_vectors"
        )
        # print(tf.shape(self._embedding))
        self._commitment_cost = commitment_cost

    def call(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        input_shape = tf.shape(inputs)
        
        # Flatten input
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
        e_latent_loss = MSE(tf.stop_gradient(quantized), inputs)
        q_latent_loss = MSE(quantized, tf.stop_gradient(inputs))
        self.loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        
        # Convert quantized back from BHWC to BCHW
        quantized = tf.transpose(quantized, perm=[0, 3, 1, 2])

        return quantized

    def get_loss(self):
        return self.loss
    