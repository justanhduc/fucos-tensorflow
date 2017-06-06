import tensorflow as tf
import numpy as np
import logging
import math


class ConvolutionalLayer:
    layers = []

    def __init__(self, input, input_shape, filter_shape, flatten=False, layer_name=None, conv_stride=(1, 1, 1, 1),
                 conv_padding='same', pool=False, pool2d=tf.nn.max_pool, pool_size=(2, 2), pool_type='same',
                 activation=tf.nn.relu, drop_out=False, drop_out_prob=0.5, batch_norm=False, batch_norm_decay=0.999):
        """

        :param input_shape: [n,h,w,c]
        :param filter_shape: [h,c,i,o]
        """
        with tf.name_scope(layer_name):
            self.input = input
            self.input_shape = input_shape
            self.filter_shape = filter_shape
            self.flatten = flatten
            self.layer_name = layer_name
            self.conv_stride = conv_stride
            self.conv_padding = conv_padding
            self.pool = pool
            self.pool2d = pool2d
            self.pool_size = pool_size
            self.pool_type = pool_type
            self.drop_out = drop_out
            self.drop_out_prob = drop_out_prob
            self.activation = activation
            self.batch_norm = batch_norm
            self.batch_norm_decay = batch_norm_decay
            self.parameters = []

            if drop_out and batch_norm:
                logging.warning('You are using both dropout and batch normalization')

            if batch_norm:
                self.bn_layer = BatchNormalizationLayer(self.get_output_shape(flatten=False),
                                                        self.batch_norm_decay, layer_name=layer_name)
            self.weights_init()
            print('@ Convolutional Layer %d' % (len(ConvolutionalLayer.layers)+1))
            print('\tinput shape: {0}'.format(self.input_shape))
            print('\tfilter shape: {0}'.format(filter_shape))
            print('\tpooling: {0} size {1}'.format(pool, pool_size))
            print('\tactivation: {0}'.format(activation))
            print('\tbatch normalization: {0}'.format(batch_norm))
            print('\tdropout: {0}'.format(drop_out))
            ConvolutionalLayer.layers.append(self)

    def weights_init(self):
        fan_in = np.prod(self.filter_shape[:3])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[:2]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_vals = tf.random_uniform_initializer(-W_bound, W_bound)
        self.parameters.append(tf.get_variable('weights', self.filter_shape, tf.float32, W_vals))
        self.parameters.append(tf.get_variable('biases', self.filter_shape[3],
                                               initializer=tf.random_normal_initializer()))

    def output(self, input=None):
        input = self.input if input is None else input
        W, b = self.parameters
        with tf.name_scope(self.layer_name):
            with tf.name_scope('convolution'):
                output_conv = tf.nn.conv2d(input, W, self.conv_stride, self.conv_padding.upper()) + b

            if self.batch_norm:
                output_conv = self.bn_layer.output(output_conv)

            if self.pool:
                with tf.name_scope('pooling'):
                    output_conv = self.pool2d(output_conv, (1, self.pool_size[0], self.pool_size[1], 1),
                                              (1, self.pool_size[0], self.pool_size[1], 1), self.pool_type.upper())

            if self.drop_out:
                with tf.name_scope('dropout'):
                    output_conv = tf.nn.dropout(output_conv, self.drop_out_prob)
            out = self.activation(output_conv)

            self.__summary('pre_activation', output_conv)
            self.__summary('activation', out)
            return tf.reshape(out, [-1, int(self.get_output_shape()[1])]) if self.flatten else out

    def get_output_shape_tensor(self, flatten=None):
        if flatten == None:
            flatten = self.flatten
        with tf.name_scope(self.layer_name):
            if self.conv_padding.lower() == 'same':
                if self.pool:
                    if self.pool_type.lower() == 'same':
                        out_shape = (self.input_shape[0],
                                     tf.to_int32(tf.ceil(tf.ceil(tf.to_float(self.input_shape[1]) /
                                                                 self.conv_stride[1]) / self.pool_size[0])),
                                     tf.to_int32(tf.ceil(tf.ceil(tf.to_float(self.input_shape[2])) /
                                                         self.conv_stride[2]) / self.pool_size[1]),
                                     self.filter_shape[3])
                    elif self.pool_type.lower() == 'valid':
                        out_shape = (self.input_shape[0],
                                     tf.to_int32(tf.floor(tf.ceil(tf.to_float(self.input_shape[1]) /
                                                                  self.conv_stride[1]) / self.pool_size[0])),
                                     tf.to_int32(
                                         tf.floor(tf.to_float(tf.ceil(tf.to_float(self.input_shape[2])) /
                                                              self.conv_stride[2]) / self.pool_size[1])),
                                     self.filter_shape[3])
                else:
                    out_shape = (self.input_shape[0],
                                 tf.to_int32(tf.ceil(tf.to_float(self.input_shape[1]) / self.conv_stride[1])),
                                 tf.to_int32(tf.ceil(tf.to_float(self.input_shape[2])) / self.conv_stride[2]),
                                 self.filter_shape[3])
            elif self.conv_padding.lower() == 'valid':
                if self.pool:
                    if self.pool_type.lower() == 'same':
                        out_shape = (self.input_shape[0],
                                     tf.to_int32(tf.ceil(np.ceil(
                                         tf.to_float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                         self.conv_stride[1])) / self.pool_size[0]),
                                     tf.to_int32(tf.ceil(np.ceil(
                                         tf.to_float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                         self.conv_stride[2])) / self.pool_size[1]),
                                     self.filter_shape[3])
                    elif self.pool_type.lower() == 'valid':
                        out_shape = (self.input_shape[0],
                                     tf.to_int32(tf.floor(np.ceil(
                                         tf.to_float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                         self.conv_stride[1])) / self.pool_size[0]),
                                     tf.to_int32(tf.floor(np.ceil(
                                         tf.to_float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                         self.conv_stride[2])) / self.pool_size[1]),
                                     self.filter_shape[3])
                else:
                    out_shape = (self.input_shape[0],
                                 tf.to_int32(
                                     tf.ceil(tf.to_float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                             self.conv_stride[1])),
                                 tf.to_int32(
                                     tf.ceil(tf.to_float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                             self.conv_stride[2])),
                                 self.filter_shape[3])
        return (out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]) if flatten else out_shape

    def get_output_shape(self, flatten=None):
        if flatten is None:
            flatten = self.flatten
        with tf.name_scope(self.layer_name):
            if self.conv_padding.lower() == 'same':
                if self.pool:
                    if self.pool_type.lower() == 'same':
                        out_shape = (self.input_shape[0],
                                     int(np.ceil(float(np.ceil(float(self.input_shape[1]) /
                                                               float(self.conv_stride[1]))) / float(self.pool_size[0]))),
                                     int(np.ceil(float(np.ceil(float(self.input_shape[2])) /
                                                       float(self.conv_stride[2])) / float(self.pool_size[1]))),
                                     self.filter_shape[3])
                    elif self.pool_type.lower() == 'valid':
                        out_shape = (self.input_shape[0],
                                     int(np.floor(float(np.ceil(float(self.input_shape[1]) /
                                                                float(self.conv_stride[1]))) /
                                                  float(self.pool_size[0]))),
                                     int(np.floor(float(np.ceil(float(self.input_shape[2])) /
                                                        float(self.conv_stride[2])) / float(self.pool_size[1]))),
                                     self.filter_shape[3])
                else:
                    out_shape = (self.input_shape[0],
                                 int(np.ceil(float(self.input_shape[1]) / float(self.conv_stride[1]))),
                                 int(np.ceil(float(self.input_shape[2])) / float(self.conv_stride[2])),
                                 self.filter_shape[3])
            elif self.conv_padding.lower() == 'valid':
                if self.pool:
                    if self.pool_type.lower() == 'same':
                        out_shape = (self.input_shape[0],
                                     int(np.ceil(float(np.ceil(
                                         float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                         float(self.conv_stride[1])))/float(self.pool_size[0]))),
                                     int(np.ceil(float(np.ceil(
                                         float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                         float(self.conv_stride[2])))/float(self.pool_size[1]))),
                                     self.filter_shape[3])
                    elif self.pool_type.lower() == 'valid':
                        out_shape = (self.input_shape[0],
                                     int(np.floor(float(np.ceil(
                                         float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                         float(self.conv_stride[1]))) / float(self.pool_size[0]))),
                                     int(np.floor(float(np.ceil(
                                         float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                         float(self.conv_stride[2]))) / float(self.pool_size[1]))),
                                     self.filter_shape[3])
                else:
                    out_shape = (self.input_shape[0],
                                 int(np.ceil(float(self.input_shape[1] - self.filter_shape[0] + 1) /
                                             float(self.conv_stride[1]))),
                                 int(np.ceil(float(self.input_shape[2] - self.filter_shape[1] + 1) /
                                             float(self.conv_stride[2]))),
                                 self.filter_shape[3])
        return (out_shape[0], np.prod(out_shape[1:])) if flatten else out_shape

    def __summary(self, name, var):
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)


class FullyConnectedLayer:
    layers = []

    def __init__(self, input, shape, layer_name=None, activation=tf.nn.relu, drop_out=False, drop_out_prob=0.5,
                 batch_norm=False, batch_norm_decay=0.9):
        """

        :param shape: [w*h*c,o]
        """
        with tf.name_scope(layer_name):
            self.input = input
            self.shape = shape
            self.layer_name = layer_name
            self.activation = activation
            self.drop_out = drop_out
            self.drop_out_prob = drop_out_prob
            self.batch_norm = batch_norm
            self.batch_norm_decay = batch_norm_decay
            if drop_out and batch_norm:
                logging.warning('You are using both dropout and batch normalization')

            self.parameters = []
            self.weights_init()

            if self.batch_norm:
                self.bn_layer = BatchNormalizationLayer(self.get_output_shape(),
                                                        self.batch_norm_decay, layer_name=layer_name)
            print('@ Fully Connected Layer %d' % (len(FullyConnectedLayer.layers)+1))
            print('\tshape: {0}'.format(shape))
            print('\tactivation: {0}'.format(activation))
            print('\tbatch normalization: {0}'.format(batch_norm))
            print('\tdropout: {0}'.format(drop_out))
            FullyConnectedLayer.layers.append(self)

    def weights_init(self):
        fan_in = self.shape[0]
        fan_out = self.shape[1]
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_vals = tf.random_uniform_initializer(-W_bound, W_bound)
        self.parameters.append(tf.get_variable('weights', self.shape, initializer=W_vals))
        self.parameters.append(tf.get_variable('biases', self.shape[1],
                                               initializer=tf.random_normal_initializer()))

    def output(self, input=None):
        input = self.input if input is None else input
        with tf.name_scope(self.layer_name):
            W, b = self.parameters
            pre_activation = tf.matmul(input, W) + b
            out = self.activation(pre_activation)
            self.__summary('pre_activaiton', pre_activation)
            self.__summary('activation', out)

            if self.drop_out:
                with tf.name_scope('dropout'):
                    out = tf.nn.dropout(out, self.drop_out_prob)
            return self.bn_layer.output(out) if self.batch_norm else out

    def get_output_shape(self):
        with tf.name_scope(self.layer_name):
            return self.shape[1]

    def __summary(self, name, var):
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)


class BatchNormalizationLayer:
    layers = []
    train_flag = tf.placeholder(dtype=tf.bool)

    def __init__(self, input_shape, decay=0.9, layer_name=None):
        with tf.name_scope(layer_name):
            self.input_shape = input_shape
            self.decay = decay
            self.layer_name = layer_name
            self.beta = tf.Variable(tf.constant(0.0, shape=[input_shape]),
                                    name='beta', trainable=True) if isinstance(input_shape, int) \
                else tf.Variable(tf.constant(0.0, shape=[input_shape[-1]]),
                                 name='beta', trainable=True)
            self.gamma = tf.Variable(tf.constant(1.0, shape=[input_shape]),
                                     name='gamma', trainable=True) if isinstance(input_shape, int) \
                else tf.Variable(tf.constant(1.0, shape=[input_shape[-1]]),
                                 name='gamma', trainable=True)
            self.ema = tf.train.ExponentialMovingAverage(self.decay)
            BatchNormalizationLayer.layers.append(self)

    def output(self, input):
        with tf.name_scope(self.layer_name):
            [mean, var] = tf.nn.moments(input, [0]) if isinstance(self.input_shape, int) \
                else tf.nn.moments(input, [0, 1, 2])

            def update_mean_var_avr():
                ema_apply = self.ema.apply([mean, var])
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(mean), tf.identity(var)

            muy, sigma = tf.cond(BatchNormalizationLayer.train_flag, update_mean_var_avr,
                                 lambda : (self.ema.average(mean), self.ema.average(var)))
            return tf.nn.batch_normalization(input, muy, sigma, self.beta, self.gamma, 1e-5)


class ConvolutionalTransposeLayer:
    layers = []

    def __init__(self, input, filter_shape, output_shape, layer_name=None, deconv_padding='SAME',
                 deconv_stride=(1, 2, 2, 1), activation=tf.nn.relu):
        with tf.name_scope(layer_name):
            self.input = input
            self.filter_shape = filter_shape
            self.output_shape = output_shape
            self.layer_name = layer_name
            self.deconv_padding = deconv_padding
            self.deconv_stride = deconv_stride
            self.activation = activation
            self.parameters = []
            self.summary = None
            self.weights_init()
            print('@ Convolutional Transpose Layer %d' % (len(ConvolutionalTransposeLayer.layers)+1))
            print('\toutput shape: {0}'.format(output_shape))
            print('\tfilter shape: {0}'.format(filter_shape))
            print('\tactivation: {0}'.format(activation))
            ConvolutionalTransposeLayer.layers.append(self)

    def weights_init(self):
        W_vals = self.get_deconv_filter()
        self.parameters.append(tf.get_variable('weights', self.filter_shape, initializer=W_vals))
        self.parameters.append(tf.get_variable('biases', self.filter_shape[2],
                                               initializer=tf.random_normal_initializer()))

    def output(self, input=None):
        input = self.input if input is None else input
        W, b = self.parameters
        pre_activation = tf.nn.conv2d_transpose(input, W, self.get_output_shape(), self.deconv_stride, 'SAME') + b
        out = self.activation(pre_activation)
        self.__summary('pre_activation', pre_activation)
        self.__summary('activation', out)
        return out

    def get_output_shape(self):
        if self.output_shape is None:
            in_shape = tf.shape(self.input)
            h = ((in_shape[1] - 1) * self.deconv_stride[1]) + 1
            w = ((in_shape[2] - 1) * self.deconv_stride[2]) + 1
            new_shape = [in_shape[0], h, w, self.filter_shape[2]]
        else:
            new_shape = [self.output_shape[0], self.output_shape[1], self.output_shape[2], self.filter_shape[2]]
        return tf.stack(new_shape)

    def get_deconv_filter(self):
        """
        This function is collected
        :param f_shape: self.filter_shape
        :return: an initializer for get_variable
        """
        width = self.filter_shape[1]
        height = self.filter_shape[0]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([self.filter_shape[0], self.filter_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(self.filter_shape)
        for i in range(self.filter_shape[2]):
            for j in range(self.filter_shape[3]):
                weights[:, :, i, j] = bilinear
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        return init

    def __summary(self, name, var):
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)
