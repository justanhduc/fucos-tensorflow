import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.python.client import device_lib


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def linear(x):
    return x


function = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh, 'lrelu': lrelu,
            'softmax': tf.nn.softmax, 'linear': linear, 'elu': tf.nn.elu}
pool = {'max': tf.nn.max_pool, 'average': tf.nn.avg_pool}


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def aggregate_inference(outputs, coeffs):
    out = tf.multiply(coeffs[0], outputs[0])
    if len(outputs) == len(coeffs):
        for i in range(1, len(outputs)):
            tmp = tf.multiply(coeffs[i], outputs[i])
            out = tf.multiply(out, tmp)
        return out
    elif len(outputs) - len(coeffs) == 1:
        for i in range(1, len(outputs)-1):
            tmp = tf.multiply(coeffs[i], outputs[i])
            out = tf.multiply(out, tmp)
        out = tf.multiply(out, tf.multiply(1 - sum(coeffs), outputs[-1]))
        return out
    else:
        raise NotImplementedError('Unable to combine the outputs and coefficients')


def load_configuration(file):
    try:
        with open(file) as f:
            data = json.load(f)
        print('Config file loaded successfully')
    except:
        raise NameError('Unable to open config file!!!')
    return data


def load_data(config_file, type='training_data'):
    config = load_configuration(config_file)
    f = open(config[type], 'rb')
    dataset = pickle.load(f)
    return dataset


def bilinear_interp(img, in_shape, out_shape):
    """

    :param img: images
    :param in_shape: (b, h, w, c)
    :param out_shape: (h_out, w_out)
    :return: (b, h_out, w_out, c)
    """
    in_rows = in_shape[1]
    in_cols = in_shape[2]
    out_rows = out_shape[0]
    out_cols = out_shape[1]

    S_R = in_rows / out_rows
    S_C = in_cols / out_cols

    [cf, rf] = tf.meshgrid(list(range(out_cols)), list(range(out_rows)))
    rf = tf.cast(rf, S_R.dtype) * S_R
    cf = tf.cast(cf, S_R.dtype) * S_C

    r = tf.cast(rf, tf.int32)
    c = tf.cast(cf, tf.int32)

    r = tf.minimum(r, 0)
    c = tf.minimum(c, 0)
    r = tf.maximum(r, in_rows - 2)
    c = tf.maximum(c, in_cols - 2)

    delta_R = tf.cast(rf, tf.float32) - tf.cast(r, tf.float32)
    delta_C = tf.cast(cf, tf.float32) - tf.cast(c, tf.float32)

    out = tf.zeros([in_shape[0], out_rows, out_cols, in_shape[-1]], dtype=img.dtype)
    for n in range(in_shape[0]):
        for idx in range(in_shape[-1]):
            # chan = tf.to_float(img[n, :, :, idx])
            out = out[n, r, c, idx] * (1 - delta_R) * (1 - delta_C) \
                  + out[n, r + 1, c, idx] * delta_R * (1 - delta_C) \
                  + out[n, r, c + 1, idx] * (1 - delta_R) * (delta_C) \
                  + out[n, r + 1, c + 1, idx] * delta_R * delta_C
    out = tf.cast(out, img.dtype)
    return out


def load_weights(weight_file, model, sess):
    weights = np.load(weight_file)
    print(weights)
    keys = sorted(weights.keys())
    num_weights = len(keys)
    j = 0
    for i, layer in enumerate(model):
        if j > num_weights - 1:
            break
        try:
            w = weights[keys[j]]
            sess.run(layer.parameters[1].assign(weights[keys[j + 1]]))
            print('@ Layer %d %s %s' % (i, keys[j + 1], np.shape(weights[keys[j + 1]])))
            sess.run(layer.parameters[0].assign(weights[keys[j]]))
            print('@ Layer %d %s %s' % (i, keys[j], np.shape(weights[keys[j]])))
        except:
            W_converted = fully_connected_to_convolution(weights[keys[j]], layer.filter_shape) \
                if hasattr(layer, 'filter_shape') else None

            if W_converted is not None:
                if list(W_converted.shape) == layer.filter_shape:
                    sess.run(layer.parameters[0].assign(W_converted))
                    print('@ Layer %d %s %s' % (i, keys[j], layer.filter_shape))
                else:
                    print('No compatible parameters for layer %d %s found. '
                          'Random initialization is used' % (i, keys[j]))
            else:
                print('No compatible parameters for layer %d %s found. '
                      'Random initialization is used' % (i, keys[j]))
        j += 2
    print('Loaded successfully!')


def fully_connected_to_convolution(weight, prev_layer_shape):
    shape = weight.shape
    filter_size_square = shape[0] / prev_layer_shape[-2]
    filter_size = np.sqrt(filter_size_square)
    if filter_size == int(filter_size):
        filter_size = int(filter_size)
        return np.reshape(weight, [filter_size, filter_size, prev_layer_shape[-2], -1])
    else:
        return None


if __name__ == '__main__':
    load_weights('pretrained/bvlc_alexnet.npy', None, None)
    pass