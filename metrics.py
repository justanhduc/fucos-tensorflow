import numpy as np
import tensorflow as tf


def GaussianCrossEntropy(y_pred, y):
    with tf.name_scope('mse'):
        return tf.reduce_mean(tf.square(y_pred - y))


def HammingDistanceCost(y_pred, y):
    with tf.name_scope('hamming'):
        return tf.reduce_mean(tf.abs(y_pred - y))


def SoftmaxCrossEntropy(y_pred, y):
    with tf.name_scope('softmaxce'):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


def MultinoulliCrossEntropy(y_pred, y):
    with tf.name_scope('sigmoidce'):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y))


def WeightedMultinoulliCrossEntropy(y_pred, y, pw):
    with tf.name_scope('weightedsigmoidce'):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, y_pred, pos_weight=pw))


def intersection_over_union(y_pred, y, dont_care=None):
    """
    :param y_pred: indicator tensor of shape [b, h, w, c]
    :param y: indicator tensor of shape [b, h, w, c]
    :return: iou value
    """
    assert y_pred.shape == y.shape, 'Predictions and groundtruths have different shapes'

    iou = []
    for c in range(y.shape[-1]):
        if dont_care is not None:
            for dc in dont_care:
                if c == dc:
                    continue
        pred = y_pred[:, :, :, c]
        gt = y[:, :, :, c]
        intersection = np.sum(np.array((gt + pred) > 1, dtype=float))
        union = np.sum(np.array((gt + pred) > 0, dtype=float))
        if union == 0:
            continue
        iou.append(intersection / union)
    return float(sum(iou)) / len(iou) * 100.


def average_class_accuracy(y_pred, y, dont_care=None):
    """
    :param y_pred: indicator tensor of shape [b, h, w, c]
    :param y: indicator tensor of shape [b, h, w, c]
    """
    assert y_pred.shape == y.shape, 'Predictions and groundtruths have different shapes'

    mean = []
    for c in range(y.shape[-1]):
        if dont_care is not None:
            for dc in dont_care:
                if c == dc:
                    continue
        pred = y_pred[:, :, :, c]
        gt = y[:, :, :, c]
        right = np.sum(np.array((gt + pred) > 1, dtype=float))
        if np.sum(np.array(gt > 0, dtype=float)) == 0:
            continue
        mean.append(right / np.sum(np.array(gt > 0, dtype=float)))
    return sum(mean) / len(mean) * 100.


def average_accuray(y_pred, y, dont_care=None):
    """
    :param y_pred: indicator tensor of shape [b, h, w, c]
    :param y: indicator tensor of shape [b, h, w, c]
    """
    assert y_pred.shape == y.shape, 'Predictions and groundtruths have different shapes'

    right = 0.
    dont_care_count = 0.
    for c in range(y.shape[-1]):
        if dont_care is not None:
            for dc in dont_care:
                if c == dc:
                    dont_care_count += np.sum(np.array(y_pred[:, :, :, c] > 1, dtype=float))
                    continue
        pred = y_pred[:, :, :, c]
        gt = y[:, :, :, c]
        right += np.sum(np.array((gt + pred) > 1, dtype=float))
    return right / (y.shape[0] * y.shape[1] * y.shape[2] - dont_care_count) * 100.


def recall(y_pred, y):
    tp = tf.reduce_sum(tf.cast((y_pred + y) > 1, tf.float32))
    fn = tf.reduce_sum(tf.cast((y_pred - y) < 0, tf.float32))
    return tp / (tp + fn)


def precision(y_pred, y):
    tp = tf.reduce_sum(tf.cast((y_pred + y) > 1, tf.float32))
    fp = tf.reduce_sum(tf.cast((y - y_pred) < 0, tf.float32))
    return tp / (tp + fp)


def f1score(y_pred, y):
    p = precision(y_pred, y)
    r = recall(y_pred, y)
    f1 = tf.cond(tf.reduce_sum(y_pred) > 0, lambda: 2 * p * r / (p + r + 1e-10), lambda: tf.constant(0.))
    return tf.cond(tf.reduce_sum(y) > 0, lambda: f1, lambda: 10)


# def spearmanrho(ypred, y, eps=1e-5):
#     error = eps * tf.random_normal([y.shape])
#     y += error       #to break tied rankings
#     return 1. - 6. * ((tf.cast(T.argsort(ypred), tf.float32) - T.cast(T.argsort(y), T.config.floatX))**2).sum() / \
#                (y.shape[0]*(y.shape[0]**2 - 1.))


def pearsoncorrelation(ypred, y):
    muy_ypred = tf.reduce_mean(ypred)
    muy_y = tf.reduce_mean(y)
    numerator = tf.reduce_sum(tf.multiply(ypred - muy_ypred, y - muy_y))
    denominator = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(ypred - muy_ypred))),
                              tf.sqrt(tf.reduce_sum(tf.square(y - muy_y)))) + 1e-10
    return numerator / denominator


def similar_derivation(y_pred, y):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    pred_dx = tf.nn.conv2d(y_pred, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    pred_dy = tf.nn.conv2d(y_pred, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')

    gt_dx = tf.nn.conv2d(y, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    gt_dy = tf.nn.conv2d(y, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')

    mse_x = GaussianCrossEntropy(pred_dx, gt_dx)
    mse_y = GaussianCrossEntropy(pred_dy, gt_dy)
    return mse_x + mse_y


if __name__ == '__main__':
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    r = pearsoncorrelation(a, b)

    sess = tf.Session()
    print(sess.run(r, feed_dict={a: np.array([1, 2, 3, 4, 5, 6, 6]), b: np.array([3, 3, 4, 5, 6, 9, 11])}))
    sess.close()
