import tensorflow as tf
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import os
import tqdm
import sys

import layers
import metrics
from utils import function, load_weights


def run_FUCOS(**kwargs):
    training_data = kwargs.get('training_data')
    validation_data = kwargs.get('validation_data')
    batchsize = kwargs.get('batchsize')
    TRAIN = kwargs.get('TRAIN', True)
    run = kwargs.get('run')

    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config_sess.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config_sess)

    #build the model
    model = []
    with tf.device('/gpu:2'):
        x = tf.placeholder(tf.float32, (None, 135, 240, 3), 'input')
        y_ = tf.placeholder(tf.float32, (None, 135, 240, 1), 'gt')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        with tf.variable_scope('conv1'):
            conv1 = layers.ConvolutionalLayer(x, [135, 240, 3], [3, 3, 3, 64])
            model.append(conv1)
        with tf.variable_scope('conv2'):
            conv2 = layers.ConvolutionalLayer(conv1.output(), conv1.get_output_shape(), [3, 3, 64, 64], pool=True)
            model.append(conv2)

        with tf.variable_scope('conv3'):
            conv3 = layers.ConvolutionalLayer(conv2.output(), conv2.get_output_shape(), [3, 3, 64, 128])
            model.append(conv3)
        with tf.variable_scope('conv4'):
            conv4 = layers.ConvolutionalLayer(conv3.output(), conv3.get_output_shape(), [3, 3, 128, 128], pool=True)
            model.append(conv4)

        with tf.variable_scope('conv5'):
            conv5 = layers.ConvolutionalLayer(conv4.output(), conv4.get_output_shape(), [3, 3, 128, 256])
            model.append(conv5)
        with tf.variable_scope('conv6'):
            conv6 = layers.ConvolutionalLayer(conv5.output(), conv5.get_output_shape(), [3, 3, 256, 256])
            model.append(conv6)
        with tf.variable_scope('conv7'):
            conv7 = layers.ConvolutionalLayer(conv6.output(), conv6.get_output_shape(), [3, 3, 256, 256], pool=True)
            model.append(conv7)

        with tf.variable_scope('conv8'):
            conv8 = layers.ConvolutionalLayer(conv7.output(), conv7.get_output_shape(), [3, 3, 256, 512])
            model.append(conv8)
        with tf.variable_scope('conv9'):
            conv9 = layers.ConvolutionalLayer(conv8.output(), conv8.get_output_shape(), [3, 3, 512, 512])
            model.append(conv9)
        with tf.variable_scope('conv10'):
            conv10 = layers.ConvolutionalLayer(conv9.output(), conv9.get_output_shape(), [3, 3, 512, 512], pool=True)
            model.append(conv10)

        with tf.variable_scope('conv11'):
            conv11 = layers.ConvolutionalLayer(conv10.output(), conv10.get_output_shape(), [3, 3, 512, 512])
            model.append(conv11)
        with tf.variable_scope('conv12'):
            conv12 = layers.ConvolutionalLayer(conv11.output(), conv11.get_output_shape(), [3, 3, 512, 512])
            model.append(conv12)
        with tf.variable_scope('conv13'):
            conv13 = layers.ConvolutionalLayer(conv12.output(), conv12.get_output_shape(), [3, 3, 512, 512], pool=True)
            model.append(conv13)

        with tf.variable_scope('conv14'):
            conv14 = layers.ConvolutionalLayer(conv13.output(), conv13.get_output_shape(), [7, 7, 512, 4096], drop_out=True,
                                               drop_out_prob=keep_prob)
            model.append(conv14)
        with tf.variable_scope('conv15'):
            conv15 = layers.ConvolutionalLayer(conv14.output(), conv14.get_output_shape(), [1, 1, 4096, 4096], drop_out=True,
                                               drop_out_prob=keep_prob)
            model.append(conv15)
        with tf.variable_scope('convtrans1'):
            deconv1 = layers.ConvolutionalTransposeLayer(conv15.output(), [4, 4, 60, 4096], None)
            model.append(deconv1)
        with tf.variable_scope('conv16'):
            conv16 = layers.ConvolutionalLayer(conv10.output(), conv10.get_output_shape(), [1, 1, 512, 60])
            model.append(conv16)
        conv16_output = conv16.output()
        sum1 = conv16_output + tf.image.resize_images(deconv1.output(), (tf.shape(conv16_output)[1],
                                                                         tf.shape(conv16_output)[2]))

        with tf.variable_scope('convtrans2'):
            deconv2 = layers.ConvolutionalTransposeLayer(sum1, [4, 4, 60, 60], None)
            model.append(deconv2)
        with tf.variable_scope('conv17'):
            conv17 = layers.ConvolutionalLayer(conv7.output(), conv7.get_output_shape(), [1, 1, 256, 60])
            model.append(conv17)
        conv17_output = conv17.output()
        sum2 = conv17_output + tf.image.resize_images(deconv2.output(), (tf.shape(conv17_output)[1],
                                                                         tf.shape(conv17_output)[2]))

        with tf.variable_scope('convtrans3'):
            deconv3 = layers.ConvolutionalTransposeLayer(sum2, [16, 16, 60, 60], None, deconv_stride=(1, 8, 8, 1))
            model.append(deconv3)

        with tf.variable_scope('conv18'):
            conv18 = layers.ConvolutionalLayer(deconv3.output(), deconv3.get_output_shape(), [1, 1, 60, 12])
            model.append(conv18)
        with tf.variable_scope('conv19'):
            conv19 = layers.ConvolutionalLayer(conv18.output(), conv18.get_output_shape_tensor(), [1, 1, 12, 1],
                                               activation=function['linear'])
            model.append(conv19)

        y_pre_activation = tf.image.resize_images(conv19.output(), (135, 240)) #resize to match the ground truth's shape
        y_pred = function['sigmoid'](y_pre_activation) #activate the output by sigmoid

        cost = metrics.MultinoulliCrossEntropy(y_pre_activation, y_) #use binary cross entropy
        var_list = tf.get_collection(tf.GraphKeys().TRAINABLE_VARIABLES)
        L2 = sum([tf.reduce_mean(tf.square(theta)) #L2 regularization
              for theta in (weight for weight in var_list if 'weights' in weight.name)])
        cost += 1e-4 * L2

        opt = tf.train.AdamOptimizer(1e-3, 0.9, 0.99, 1e-8).minimize(cost, var_list=var_list) #ADAM optimization
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_pred >= 0.5, tf.uint8), tf.cast(y_, tf.uint8)), tf.float32))
        saver = tf.train.Saver()

        if TRAIN:
            tf.Operation.run(tf.global_variables_initializer())
            print('Loading VGG16 weights...')
            load_weights('pretrained/vgg16_weights.npz', model, sess) #load pretrained VGG16 weights

            best_valid_accuracy = 0.
            best_valid_loss = np.inf
            best_epoch = 0
            epoch = 0
            vote_to_terminate = 0
            done_looping = False
            print('TRAINING...')
            start_training_time = time.time()
            while epoch < 200 and not done_looping:
                epoch += 1
                num_iter_training = int(training_data[0].shape[0] / batchsize)
                losses_train = 0.
                accuracies_train = 0.
                start_batch_time = time.time()
                print('Epoch %d...' % epoch)
                batch = next_batch(training_data, batchsize) #training
                for b in batch:
                    fd = {x: b[0], y_: b[1], keep_prob: 0.1}
                    _, a, l = sess.run([opt, accuracy, cost], feed_dict=fd)
                    assert not np.isnan(l), 'Train failed with loss being NaN'
                    losses_train += l
                    accuracies_train += a

                print('\ttraining loss: %s' % (losses_train / num_iter_training))
                print('\ttraining accuracy: %s' % (accuracies_train / num_iter_training))
                print('\tepoch %d took %.2f hours' % (epoch, (time.time() - start_batch_time) / 3600.))

                num_iter_valid = int(validation_data[0].shape[0] / batchsize)
                losses_valid = 0.
                accuracies_valid = 0.
                start_valid_time = time.time()
                batch = next_batch(validation_data, batchsize) #validation
                for b in batch:
                    fd = {x: b[0], y_: b[1], keep_prob: 1}
                    l, a = sess.run([cost, accuracy], feed_dict=fd)
                    losses_valid += l
                    accuracies_valid += a
                avr_acc_valid = accuracies_valid / num_iter_valid
                losses_valid /= num_iter_valid

                print('\tvalidation took %.2f hours' % ((time.time() - start_valid_time) / 3600.))
                print('\tvalidation loss: %s' % losses_valid)
                print('\tvalidation accuracy: %s' % avr_acc_valid)

                if losses_valid < best_valid_loss:
                    best_valid_loss = losses_valid
                    best_epoch = epoch
                    vote_to_terminate = 0
                    print('\tbest validation loss achieved: %.4f' % best_valid_loss)
                    save_path = saver.save(sess, run)
                    print("\tmodel saved in file: %s" % save_path)
                else:
                    vote_to_terminate += 1

                if vote_to_terminate > 30:
                    done_looping = True
            print('Training ends after %.2f hours' % ((time.time() - start_training_time) / 3600.))
            print('\tbest validation accuracy: %.2f' % best_valid_accuracy)
            print('Training the model using all data available...')
            total_training_data = (np.concatenate((training_data[0], validation_data[0])),
                                   np.concatenate((training_data[1], validation_data[1])))
            for i in range(best_epoch):
                num_iter_training = int(total_training_data[0].shape[0] / batchsize)
                losses_train = 0.
                start_batch_time = time.time()
                print('Epoch %d...' % (i+1))
                batch = next_batch(total_training_data, batchsize) #training
                for b in batch:
                    fd = {x: b[0], y_: b[1], keep_prob: 0.1}
                    _, _, l = sess.run([opt, accuracy, cost], feed_dict=fd)
                    assert not np.isnan(l), 'Train failed with loss being NaN'
                    losses_train += l

                print('\ttraining loss: %s' % (losses_train / num_iter_training))
                print('\tepoch %d took %.2f hours' % (i+1, (time.time() - start_batch_time) / 3600.))

        else: #testing
            path = kwargs.get('testing_path')
            isfolder = kwargs.get('isfolder')

            image_list = [path + '/' + f for f in os.listdir(path) if f.endswith('.jpg')] if isfolder else [path]
            saver.restore(sess, tf.train.latest_checkpoint(run))
            print('Checkpoint restored...')
            print('Testing %d images...' % len(image_list))
            images = []
            predictions = []
            time.sleep(0.1)
            for i in tqdm.tqdm(range(len(image_list)), unit='images'):
                img = misc.imread(image_list[i])
                if len(img.shape) < 3:
                    continue
                img = np.reshape(misc.imresize(img, (135, 240)), (1, 135, 240, 3)) / 255.
                fd = {x: img, keep_prob: 1}
                pred = sess.run(y_pred, feed_dict=fd)
                images.append(img)
                predictions.append(pred)
            time.sleep(0.1)
            print('Testing finished!')

            for i in range(len(images)):
                plt.figure(1)
                image = np.reshape(images[i], (135, 240, 3))
                sal = np.reshape(predictions[i], (135, 240))
                sal = sal * (sal > np.percentile(sal, 95))
                sal = gaussian_filter(sal, sigma=10)
                sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))
                plt.subplot(211)
                plt.imshow(image)
                plt.subplot(212)
                plt.imshow(sal, cmap='gray')
                plt.show()



def load_data(data_file):
    return pickle.load(open(data_file, 'rb'))


def next_batch(data, batchsize, ground_truth=True):
    if ground_truth:
        x, y = data
    else:
        x = data
    index_shuf = list(range(x.shape[0]))
    shuffle(index_shuf)
    x = x[index_shuf]
    if ground_truth:
        y = y[index_shuf]
    num_batches = int(x.shape[0] / batchsize)
    for i in range(num_batches):
        yield (x[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]) if ground_truth \
            else x[i*batchsize:(i+1)*batchsize]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise NotImplementedError
    else:
        training_data = load_data('training.pkl')
        validation_data = load_data('validation.pkl')

        phase = sys.argv[1]
        checkpoint_fol = sys.argv[2]
        if phase.lower() == 'train':
            kwargs = {'training_data': training_data, 'validation_data': validation_data, 'batchsize': 10,
                      'run': checkpoint_fol, 'TRAIN': True}
            run_FUCOS(**kwargs)
        elif phase.lower() == 'test':
            if len(sys.argv) < 3:
                raise SyntaxError
            elif len(sys.argv) == 4:
                is_folder = True
            elif len(sys.argv) == 5:
                is_folder = sys.argv[4]
            else:
                raise SyntaxError
            testing_path = sys.argv[3]
            kwargs = {'training_data': training_data, 'validation_data': validation_data, 'batchsize': 10,
                      'run': checkpoint_fol, 'TRAIN': False, 'testing_path': testing_path, 'isfolder': is_folder}
            run_FUCOS(**kwargs)
        else:
            raise NotImplementedError
