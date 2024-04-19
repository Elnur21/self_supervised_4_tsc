import tensorflow as tf
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class LITE:

    def __init__(self,input_shape,n_dim,batch_size=64,
        n_filters=32,
        kernel_size=41,
        n_epochs=1500,
        verbose=True,
        use_custom_filters=True,
        use_dilation=True,
        use_multiplexing=True,
    ):

        self.input_shape = input_shape
        self.n_classes = n_dim

        self.verbose = verbose

        self.n_filters = n_filters

        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing

        self.kernel_size = kernel_size - 1

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.build_model()

    def hybird_layer(
        self, input_tensor, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64]
    ):  
        # print(self.input_shape)
        input_tensor = tf.keras.layers.Reshape(target_shape=(self.input_shape[0],1),name='fff_1')(input_tensor)
        # print(input_tensor)
        conv_list = []

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1

            conv = tf.keras.layers.Conv1D(
                filters=input_channels,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-increasse-"
                + str(self.keep_track)
                + "-"
                + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1

            conv = tf.keras.layers.Conv1D(
                filters=input_channels,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-decrease-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)
            conv_list.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes[1:]:

            filter_ = np.zeros(
                shape=(kernel_size + kernel_size // 2, input_channels, 1)
            )

            xmash = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
                (-1, 1, 1)
            )

            filter_left = xmash**2
            filter_right = filter_left[::-1]

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            conv = tf.keras.layers.Conv1D(
                filters=input_channels,
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-peeks-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)

            self.keep_track += 1

        hybird_layer = tf.keras.layers.Concatenate(axis=2)(conv_list)
        hybird_layer = tf.keras.layers.Activation(activation="relu")(hybird_layer)

        return hybird_layer

    def _inception_module(
        self,
        input_tensor,
        dilation_rate,
        stride=1,
        activation="linear",
        use_hybird_layer=False,
        use_multiplexing=True,
    ):

        input_inception = input_tensor

        if not use_multiplexing:

            n_convs = 1
            n_filters = self.n_filters * 3

        else:
            n_convs = 3
            n_filters = self.n_filters

        kernel_size_s = [self.kernel_size // (2**i) for i in range(n_convs)]

        conv_list = []
        reshape = tf.keras.layers.Reshape(target_shape=(self.input_shape[0],1),name='fff')(input_inception)
        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=False,
                )(reshape)
            )

        if use_hybird_layer:
            self.hybird = self.hybird_layer(
                input_tensor=input_tensor, input_channels=input_tensor.shape[-1]
            )
            conv_list.append(self.hybird)

        if len(conv_list) > 1:
            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        else:
            x = conv_list[0]

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)

        return x

    def _fcn_module(
        self,
        input_tensor,
        kernel_size,
        dilation_rate,
        n_filters,
        stride=1,
        activation="relu",
    ):

        x = tf.keras.layers.SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilation_rate,
            use_bias=False,
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def build_model(self):

        self.keep_track = 0

        input_shape = self.input_shape

        input_layer = tf.keras.layers.Input(input_shape)
        reshape_layer = tf.keras.layers.Reshape(target_shape=self.input_shape)(
            input_layer
        )

        inception = self._inception_module(
            input_tensor=reshape_layer,
            dilation_rate=1,
            use_hybird_layer=self.use_custom_filters,
        )

        self.kernel_size //= 2

        input_tensor = inception

        dilation_rate = 1

        for i in range(2):

            if self.use_dilation:
                dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=self.kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(gap)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

      