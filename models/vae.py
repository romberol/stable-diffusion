import tensorflow as tf
import tensorflow_addons as tfa


class SampleLayer(tf.keras.layers.Layer):
    """
    Implements a sampling layer for use in variational autoencoder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Splits the tensor into the means and log variances components,
        performs the sampling operation, calculates the Kullback-Leibler (KL) loss, 
        and adds it to the layer's losses

        Parameters:
        - inputs (Tensor): means and log of variances of shape (batch_size, height, width, 2 * num_latent_dims)

        Returns:
        - Tensor: Samples from distribution with received means and variances 
        """
        mean, log_var = tf.split(inputs, 2, axis=3)
        epsilon = tf.random.normal(tf.shape(log_var))
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=[1, 2, 3])
        self.add_loss(tf.reduce_mean(kl_loss))
        return epsilon * tf.exp(log_var / 2) + mean


class ResNet(tf.keras.layers.Layer):
    """
    Class that implements a residual block for use in an autoencoder.
    """

    def __init__(self, inp_channels, out_channels, **kwargs):
        """
        - inp_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        """
        super().__init__(**kwargs)

        self.norm1 = tfa.layers.GroupNormalization(groups=inp_channels, epsilon=1e-5)
        self.norm2 = tfa.layers.GroupNormalization(groups=out_channels, epsilon=1e-5)
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, kernel_initializer="he_normal", padding="same")
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, kernel_initializer="he_normal", padding="same")

        self.skip = tf.keras.layers.Conv2D(out_channels, 1, kernel_initializer="he_normal",
                                           padding="same") if inp_channels != out_channels \
            else tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs):
        """
        Forward pass of the residual block.

        Parameters:
        - inputs (Tensor): The input tensor to the residual block.

        Returns:
        - Tensor: The output tensor of the residual block.
        """
        Z = self.norm1(inputs)
        Z = tf.keras.activations.swish(Z)
        Z = self.conv1(Z)

        Z = self.norm2(Z)
        Z = tf.keras.activations.swish(Z)
        Z = self.conv2(Z)

        return self.skip(inputs) + Z


class AttnBlock(tf.keras.layers.Layer):
    """
    Self Attention block
    """

    def __init__(self, channels):
        """
        - channels (int): The number of channels.
        """
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(groups=channels, epsilon=1e-5)
        self.q = tf.keras.layers.Conv2D(channels, 1, activation=None, kernel_initializer="he_normal")
        self.k = tf.keras.layers.Conv2D(channels, 1, activation=None, kernel_initializer="he_normal")
        self.v = tf.keras.layers.Conv2D(channels, 1, activation=None, kernel_initializer="he_normal")
        self.proj_out = tf.keras.layers.Conv2D(channels, 1, activation=None, kernel_initializer="he_normal")
        self.scale = channels ** -0.5

    def call(self, x):
        """
        Forward pass of the attention block.

        Parameters:
        - x (Tensor): The input tensor to the attention block.

        Returns:
        - Tensor: The output tensor of the attention block.
        """
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b = tf.shape(q)[0]
        c = tf.shape(q)[1]
        h = tf.shape(q)[2]
        w = tf.shape(q)[3]

        q = tf.reshape(q, (b, c, h * w))
        k = tf.reshape(k, (b, c, h * w))
        v = tf.reshape(v, (b, c, h * w))
        attn = tf.linalg.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=2)
        out = tf.linalg.matmul(attn, v)
        out = tf.reshape(out, (b, c, h, w))
        out = self.proj_out(out)
        return x + out


class Encoder(tf.keras.Model):
    """
    Encoder part of variational autoencoder.
    """

    def __init__(self, channels, channel_mults, n_resnet, z_channels, **kwargs):
        """
        - channels (int): The number of channels for the initial convolutional layer.
        - channel_mults (list): A list of channel multipliers for each downsampling block.
        - n_resnet (int): The number of residual blocks in each downsampling block.
        - z_channels (int): The number of channels for the final encoded representation.
        """
        super().__init__(**kwargs)

        self.inp_conv = tf.keras.layers.Conv2D(channels, 3, kernel_initializer="he_normal", padding="same")

        self.down = tf.keras.Sequential()

        channels_list = [mult * channels for mult in channel_mults]

        for i in range(len(channel_mults) - 1):
            for _ in range(n_resnet):
                self.down.add(ResNet(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

            # downsample
            self.down.add(tf.keras.layers.Conv2D(channels, 3, strides=2, padding='same',
                                                 kernel_initializer="he_normal"))

        self.mid1 = ResNet(channels, channels)
        self.attn = AttnBlock(channels)
        self.mid2 = ResNet(channels, channels)

        self.norm_out = tfa.layers.GroupNormalization(groups=channels, epsilon=1e-5)
        self.conv_out = tf.keras.layers.Conv2D(2 * z_channels, 3, kernel_initializer="he_normal", padding="same")

    def call(self, inputs):
        """
        Forward pass of the encoder.

        Parameters:
        - inputs (Tensor): The input tensor to the encoder.

        Returns:
        - Tensor: The output tensor of the encoder.
        """
        Z = self.inp_conv(inputs)

        Z = self.down(Z)

        Z = self.mid1(Z)
        Z = self.attn(Z)
        Z = self.mid2(Z)

        Z = self.norm_out(Z)
        Z = tf.keras.activations.swish(Z)

        Z = self.conv_out(Z)

        return Z


class Decoder(tf.keras.Model):
    """
    Decoder part of variational autoencoder.
    """

    def __init__(self, channels, channel_mults, n_resnet, out_channels, **kwargs):
        """
        - channels (int): The number of channels for the initial convolutional layer.
        - channel_mults (list): A list of channel multipliers for each upsampling block.
        - n_resnet (int): The number of residual blocks in each upsampling block.
        - out_channels (int): The number of channels for the final reconstructed output.
        """
        super().__init__(**kwargs)

        channels_list = [mult * channels for mult in channel_mults]
        channels = channels_list[-1]

        self.inp_conv = tf.keras.layers.Conv2D(channels, 3, kernel_initializer="he_normal", padding="same")

        self.mid1 = ResNet(channels, channels)
        self.attn = AttnBlock(channels)
        self.mid2 = ResNet(channels, channels)

        self.up = tf.keras.Sequential()

        for i in reversed(range(len(channel_mults) - 1)):
            for _ in range(n_resnet):
                self.up.add(ResNet(channels, channels_list[i]))
                channels = channels_list[i]

            # upsample
            self.up.add(tf.keras.layers.Conv2DTranspose(channels, 3, padding="same", strides=2))
            self.up.add(tf.keras.layers.Conv2D(channels, 3, padding="same", kernel_initializer="he_normal"))

        self.norm_out = tfa.layers.GroupNormalization(groups=channels, epsilon=1e-5)
        self.conv_out = tf.keras.layers.Conv2D(out_channels, 3, kernel_initializer="glorot_normal",
                                               activation="sigmoid", padding="same")

    def call(self, inputs):
        """
        Forward pass of the decoder.

        Parameters:
        - inputs (Tensor): The input tensor to the decoder.

        Returns:
        - Tensor: The output tensor of the decoder.
        """
        Z = self.inp_conv(inputs)

        Z = self.mid1(Z)
        Z = self.attn(Z)
        Z = self.mid2(Z)

        Z = self.up(Z)

        Z = self.norm_out(Z)
        Z = tf.keras.activations.swish(Z)
        Z = self.conv_out(Z)

        return Z


class Autoencoder(tf.keras.Model):
    """
    Variational autoencoder
    """

    def __init__(self, encoder, decoder, emb_channels, z_channels, **kwargs):
        """
        - encoder (tf.keras.Model): The encoder model.
        - decoder (tf.keras.Model): The decoder model.
        - emb_channels (int): The number of channels for the intermediate embedding.
        - z_channels (int): The number of channels for the final embedding space.
        """
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.quantization = tf.keras.layers.Conv2D(2 * emb_channels, 1, padding="same", kernel_initializer="he_normal")
        self.post_quantization = tf.keras.layers.Conv2D(z_channels, 1, padding="same", kernel_initializer="he_normal")
        self.sample = SampleLayer()

    def encode(self, inputs):
        """
        Encodes the input tensor into a latent representation.

        Parameters:
        - inputs (Tensor): The input tensor to encode.

        Returns:
        - Tensor: The encoded latent representation.
        """
        Z = self.encoder(inputs)
        Z = self.quantization(Z)
        return self.sample(Z)

    def decode(self, inputs):
        """
        Decodes the latent representation into a reconstructed output.

        Parameters:
        - inputs (Tensor): The input tensor to decode.

        Returns:
        - Tensor: The reconstructed output.
        """
        Z = self.post_quantization(inputs)
        return self.decoder(Z)

    def call(self, inputs):
        """
        Forward pass of the autoencoder.

        Parameters:
        - inputs (Tensor): The input tensor to the autoencoder.

        Returns:
        - Tensor: The reconstructed output.
        """
        Z = self.encode(inputs)
        return self.decode(Z)
