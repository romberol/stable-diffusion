import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class TimeEncoding(tf.keras.layers.Layer):
    """
    Sine/cosine positional embeddings
    """
    def __init__(self, T, embed_size, dtype=tf.float32, **kwargs):
        """
        - T (int): The maximum time value to encode (number of timesteps for diffusion model).
        - embed_size (int): The size of the embedding vector.
        - dtype (tf.dtypes.DType): The data type of the layer's computations (default: tf.float32).
        """
        super().__init__(dtype=dtype, **kwargs)
        self.T = T
        self.embed_size = embed_size
        p, i = np.meshgrid(np.arange(T + 1), 2 * np.arange(embed_size // 2))
        t_emb = np.empty((T + 1, embed_size))
        t_emb[:, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        t_emb[:, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.time_encodings = tf.constant(t_emb.astype(self.dtype))

    def call(self, inputs):
        """
        Forward pass of the TimeEncoding layer.

        Parameters:
        - inputs (Tensor): The input tensor to encode.

        Returns:
        - Tensor: The time-encoded output tensor.
        """
        return tf.gather(self.time_encodings, inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "T": self.T,
            "embed_size": self.embed_size
        })
        return config

class TimeResNet(tf.keras.layers.Layer):
    """
    Implements a residual block with time embedding.
    """
    def __init__(self, inp_channels, out_channels):
        """
        - inp_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        """
        super().__init__()
        self.norm1 = tfa.layers.GroupNormalization(groups=inp_channels, epsilon=1e-5)
        self.norm2 = tfa.layers.GroupNormalization(groups=out_channels, epsilon=1e-5)
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, kernel_initializer="he_normal", padding="same")
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, kernel_initializer="he_normal", padding="same")
        
        self.time_embed = tf.keras.layers.Dense(out_channels, kernel_initializer="he_normal")
        
        self.skip = tf.keras.layers.Conv2D(out_channels, 1, kernel_initializer="he_normal", padding="same") if inp_channels != out_channels \
                    else tf.keras.layers.Lambda(lambda x: x)
        
    def call(self, inputs):
        """
        Forward pass of the TimeResNet layer.

        Parameters:
        - inputs (tuple): A tuple containing two tensors: x (input tensor) and time (time tensor).

        Returns:
        - Tensor: The output tensor.
        """
        x, time = inputs
        
        Z = self.norm1(x)
        Z = tf.keras.activations.swish(Z)
        Z = self.conv1(Z)
        
        time = tf.keras.activations.swish(time)
        time_emb = self.time_embed(time)
        
        Z = time_emb[:, tf.newaxis, tf.newaxis, :] + Z

        Z = self.norm2(Z)
        Z = tf.keras.activations.swish(Z)
        Z = self.conv2(Z)

        return self.skip(x) + Z
    
    
class UNetAttention(tf.keras.layers.Layer):
    """
    Cross attention mechanism in a U-Net architecture.
    """
    def __init__(self, dim):
        """
        - dim (int): The number of filters.
        """
        super().__init__()
        self.conv_g = tf.keras.layers.Conv2D(dim, 1)
        self.conv_x = tf.keras.layers.Conv2D(dim, 1, strides=2)

        self.conv_out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
        self.upsample = tf.keras.layers.UpSampling2D(2)
        self.multiply = tf.keras.layers.Multiply()
        
    def call(self, inputs):
        """
        Forward pass of the UNetAttention layer.

        Parameters:
        - inputs (tuple): A tuple containing two tensors: 
            x_inp (input tensor with dim filters) 
            g_inp (input tensor with dim/2 filters).

        Returns:
        - Tensor: The output tensor.
        """    
        x_inp, g_inp = inputs

        g = self.conv_g(g_inp)
        x = self.conv_x(x_inp)

        g = tf.keras.layers.Add()([g, x])
        g = tf.keras.activations.relu(g)
        g = self.conv_out(g)

        g = self.upsample(g)

        out = self.multiply([x_inp, g])
        return out
    
def get_diff_model(start_dim, out_dim, channels, input_shape, T, embed_size):
    """
    Constructs the diffusion model.

    Parameters:
    - start_dim (int): Number of filters in the initial layer.
    - out_dim (int): Number of filters in the output layer.
    - channels (list): List of integers specifying the number of filters in each layer.
    - input_shape (tuple): Shape of the input tensor.
    - T (int): Length of the time sequence.
    - embed_size (int): Size of the time embedding.

    Returns:
    - tf.keras.Model: The constructed diffusion model.
    """
    X_noisy = tf.keras.layers.Input(shape=input_shape, name="X_noisy")
    time_input = tf.keras.layers.Input(shape=[], dtype=tf.int32, name="time")
    time_enc = TimeEncoding(T, embed_size)(time_input)
    
    inp = tf.keras.Sequential([
        tf.keras.layers.Conv2D(start_dim, 3, kernel_initializer="he_normal", padding="same"),
        tfa.layers.GroupNormalization(groups=start_dim, epsilon=1e-5),
        tf.keras.layers.Activation("swish")
    ])
    
    out = tf.keras.Sequential([
        tfa.layers.GroupNormalization(groups=channels[0], epsilon=1e-5),
        tf.keras.layers.Activation("swish"),
        tf.keras.layers.Conv2D(out_dim, 3, kernel_initializer="he_normal", padding="same")
    ])

    Z = inp(X_noisy)
    
    time_embed = tf.keras.Sequential([
        tf.keras.layers.Dense(start_dim * 4),
        tf.keras.layers.Activation("swish"),
        tf.keras.layers.Dense(start_dim * 4)
    ])

    time = time_embed(time_enc)

    skip = Z
    cross_skips = []
    last_dim = start_dim
    
    for i in range(len(channels)):
        dim = channels[i]
        Z = TimeResNet(last_dim, dim)([Z, time])
        if i != len(channels) -1 :
            cross_skips.append(Z)
            Z = tf.keras.layers.Conv2D(dim, 3, strides=2, padding="same")(Z)
            last_dim = dim
        
        else:
            attention = UNetAttention(last_dim)([cross_skips.pop(), Z])
            Z = tf.keras.layers.UpSampling2D(2)(Z)
            Z = tf.keras.layers.concatenate([Z, attention], axis=-1)
            last_dim += dim
     
    for i in reversed(range(len(channels)-1)):
        dim = channels[i]
        Z = TimeResNet(last_dim, dim)([Z, time])
        
        if i != 0:
            attention = UNetAttention(dim)([cross_skips.pop(), Z])
            Z = tf.keras.layers.UpSampling2D(2)(Z)
            Z = tf.keras.layers.concatenate([Z, attention], axis=-1)
            last_dim = dim + channels[i-1]
    
    outputs = out(Z)
    
    return tf.keras.Model(inputs=[X_noisy, time_input], outputs=[outputs])