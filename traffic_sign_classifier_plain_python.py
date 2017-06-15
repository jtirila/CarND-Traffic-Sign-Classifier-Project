# Load pickled data
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from datetime import datetime
import os
import re

# These are for image transformations
import cv2
import csv

from random import random


# For testing purposes:
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Some global variables

FIRST_CONVO_WEIGHTS_NAME = 'first_convo_weights'
FIRST_CONVO_WEIGHTS_PATTERN = re.compile(FIRST_CONVO_WEIGHTS_NAME + '.*')
SECOND_CONVO_WEIGHTS_NAME = 'second_convo_weights'
SECOND_CONVO_WEIGHTS_PATTERN = re.compile(SECOND_CONVO_WEIGHTS_NAME + '.*')

TRAINING_FILE = '/Users/jmt/Data/german-traffic-signs/train.p'
VALIDATION_FILE = '/Users/jmt/Data/german-traffic-signs/valid.p'
TESTING_FILE = '/Users/jmt/Data/german-traffic-signs/test.p'
LABEL_FILE = 'signnames.csv'


LEARNING_RATE = 0.0014
EPOCHS = 50
BATCH_SIZE = 128

WEB_FILENAMES_ORIGINAL = ['5-speed-limit-80-km-h-cropped.png', '30-snow-cropped.png', '31-wild-animals-passing-cropped.png', '38-keep-right-cropped.png', '17-no-entry-cropped.png']
WEB_FILENAMES = ['5-speed-limit-80-km-h-cropped-3.png', '30-snow-cropped-3.png',
                 '31-wild-animals-passing-cropped-3.png', '38-keep-right-cropped-3.png',
                 '17-no-entry-cropped-3.png']


def testing_pipeline():
    """Load test data and previously saved model, print statistics."""
    test_features, test_labels = _load_real_validation_data()

    print("Number of examples in test set:", len(test_features))

    # TODO: remove?
    # normalized_features = []
    # for img in test_features:
    #     normalized_features.append(_convert_color_image(img))

    test_features = [_convert_color_image(img) for img in test_features]
    # test_features = np.array(list(map(_convert_color_image, test_features)))
    load_and_evaluate_model(test_features, test_labels)


def testing_pipeline_web(visualize=True):
    features, labels, orig_label_stats, preprocessed_label_stats = load_and_preprocess_web_images(visualize=visualize)
    _print_test_data_basic_summary(features, labels, orig_label_stats, preprocessed_label_stats)
    load_and_evaluate_model(features, labels, "web")


def training_pipeline(visualize=True, mnist_test=False):
    """Load training data, define and train the model, then save to disk. """
    if mnist_test:
        train_features, train_labels, valid_features, valid_labels = _load_test_data()
    else:
        train, valid, test = _load_previously_saved_data()
        train_features, train_labels = train['features'], train['labels']
        valid_features, valid_labels = valid['features'], valid['labels']
        train_features, train_labels = shuffle(train_features, train_labels)
        valid_features, valid_labels = shuffle(valid_features, valid_labels)

    if visualize:
        # Visualize training data first without preprocessing
        _visualize_data(train_features, train_labels)

    if not mnist_test:
        train_features, train_labels, valid_features, \
        valid_labels, orig_label_stats_train, preprocessed_label_stats_train, \
        orig_label_stats_valid, preprocessed_label_stats_valid = _preprocess_train_test_data(
            train_features, train_labels, valid_features, valid_labels)
        _print_test_data_basic_summary(train_features, train_labels, orig_label_stats_train, preprocessed_label_stats_train)
        _print_test_data_basic_summary(valid_features, valid_labels, orig_label_stats_valid, preprocessed_label_stats_valid)

    if visualize:
        # Now visualize after preprocessing
        _visualize_data(train_features, train_labels, split=True)

    train_features, train_labels = shuffle(train_features, train_labels)

    # Work with the actual model begins
    network = _define_model_architecture()

    # Use web images also
    own_images, own_labels, orig_label_stats, preprocessed_label_stats = load_and_preprocess_web_images(visualize=visualize)
    _print_test_data_basic_summary(own_images, own_labels, orig_label_stats, preprocessed_label_stats)
    _train_network_and_save_params(network, train_features, train_labels, valid_features, valid_labels,
                                   own_images, own_labels)


def _load_real_validation_data():
    train, valid, test = _load_previously_saved_data()
    x_test, y_test = test['features'], test['labels']
    x_test, y_test = shuffle(x_test, y_test)
    return x_test, y_test


def _load_test_validation_data():
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    x_test, y_test = mnist.test.images, mnist.test.labels
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test, y_test = shuffle(x_test, y_test)
    return x_test, y_test


def _augment_image_data(image):
     """FIXME get image data as input, return same structure but an augmented version."""


def _load_test_data():
    """Loads the mnist character data. This is a short-term placeholder to get building something before getting the
    real data set prepared."""
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    # Pad images with 0s
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_validation = np.pad(x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    # Shuffle the data
    x_train, y_train, = \
        shuffle(x_train, y_train)
    x_validation, y_validation = \
        shuffle(x_validation, y_validation)
    return x_train, y_train, x_validation, y_validation


def _load_previously_saved_data():
    """Loads the data.
    :return: A tuple containing the test, validation and test sets of data"""

    train = _load_data_file(TRAINING_FILE)
    valid = _load_data_file(VALIDATION_FILE)
    test = _load_data_file(TESTING_FILE)
    return train, valid, test


def _load_data_file(path):
    """Loads a single file.
    :param path: A path, pointing to a pickled data file.
    :return: a depickled data file"""
    with open(path, mode='rb') as f:
        return pickle.load(f)


def _print_test_data_basic_summary(features, labels, orig_label_stats, preprocessed_label_stats):
    assert len(features) == len(labels)
    print("Number or examples: {}".format(len(features)))
    print("Original label stats: {}".format(orig_label_stats))
    print("Maximum number of examples in a class: {}".format(max(orig_label_stats.values())))
    print("Label stats after preprocessing: {}".format(preprocessed_label_stats))


def _print_training_data_basic_summary(train_features, train_labels, valid_features, valid_labels):
    # ### Replace each question mark with the appropriate value.
    # ### Use python, pandas or numpy methods rather than hard coding the results

    # return
    # Number of training examples
    n_train = len(train_features)
    n_valid = len(valid_features)


    # Shape of an traffic sign image?
    shape = train_features[0].shape
    image_shape = "{} x {} x {}".format(shape[0], shape[1], shape[2])

    n_classes = len(set(train_labels))

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Image data shape = ", image_shape)
    print("Number of classes =", n_classes)


def _load_web_images_first_time():
    # Loads the images in BRG mode.
    images = np.array(list(map(lambda x: cv2.imread(os.path.join('web-images', x)), WEB_FILENAMES)))
    images = np.array(
        [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    )
    labels = [x.split("-")[0] for x in WEB_FILENAMES]
    return images, labels


def _load_web_images_from_pickle(pickle_file):
    return pickle.load(pickle_file)


def _resize_images_to_32_x_32(images):
    return list(map(_resize_image_to_32_x_32, images))


def _resize_image_to_32_x_32(image):
    """Resizes an image to 32 x 32 pixels."""
    return cv2.resize(image, (32, 32))


def _visualize_data(features, labels, split=False):
    label_dict = {}
    with open('signnames.csv', 'r') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            label_dict[row['ClassId']] = row['SignName']

    halfway = len(features) // 2

    if split:
        for i in range(1, 9):
            plt.subplot(4, 4 , i)
            plt.imshow(features[i - 1])
            plt.axis('off')
            plt.title("Ind {} - {}: {}".format(i - 1, labels[i - 1], label_dict[str(labels[i - 1])][:10]))
        for i in range(1, 9):
            plt.subplot(4,4,i + 8)
            plt.imshow(features[halfway + i - 1])
            plt.axis('off')
            plt.title("Ind {} - {}: {}".format(halfway + i - 1, labels[halfway + i - 1], label_dict[str(labels[halfway + i - 1])][:10]))
        plt.show()
    elif len(features) > 5:
        # Plot the 12 first images
        for i in range(1, 9):
            plt.subplot(4, 3, i)
            if len(features) > i + 1:
                plt.imshow(features[i - 1])
                plt.axis('off')
                plt.title("Ind {} - {}: {}".format(i - 1, labels[i - 1], label_dict[str(labels[i - 1])][:10]))
        plt.show()
    elif len(features) == 5:
        for i in range(1, 6):
            plt.subplot(2, 4, i)
            plt.imshow(features[i - 1])
            plt.axis('off')
            plt.title("Ind {} - {}: {}".format(i - 1, labels[i - 1], label_dict[str(labels[i - 1])][:10]))
        plt.show()
    else:
        raise Exception("Wrong image number specification!")


def _preprocess_data(features, labels):
    normalized_features = np.array([])

    # halfway_train = len(X_train) // 2
    # train_1st_half_copy = X_train[:halfway_train]
    # train_2st_half_copy = X_train[halfway_train:]

    # labels_train_1st_half = y_train[:halfway_train]
    # labels_train_2st_half = y_train[halfway_train:]

    # Find the number of occurrences of each of the labels in the training data:

    orig_label_stats = {ind: list(labels).count(ind) for ind in set(labels)}
    max_num_labels = max(orig_label_stats.values())

    for label, num in orig_label_stats.items():
        # Find all images with this label
        imgs = np.array([img_label[0] for img_label in zip(features, labels) if img_label[1] == label])
        coeff = max_num_labels / orig_label_stats[label] - 1
        coeff_int = int(np.floor(coeff))
        original_length = len(imgs)
        if coeff_int > 1:
            imgs_repeated = [img for img in imgs for _ in range(coeff_int)]
        else:
            imgs_repeated = np.array([])
        max_ind = int((coeff - coeff_int) * original_length)
        if max_ind > 0:
            if len(imgs_repeated) > 0:
                imgs_repeated = np.concatenate((imgs_repeated, imgs[:max_ind]))
            else:
                imgs_repeated = imgs[:max_ind]

        rotation_angles = [18 * (random() - 0.5) for _ in range(len(imgs_repeated))]
        scale_coeffs = [1 + 0.2 * (random() - 0.2) for _ in range(len(imgs_repeated))]

        for ind, img in enumerate(imgs_repeated):
            matr = cv2.getRotationMatrix2D((16, 16), rotation_angles[ind], scale_coeffs[ind])
            imgs_repeated[ind] = cv2.warpAffine(img, matr, (32, 32))

        if len(imgs_repeated) > 0:
            new_labels = np.tile(label, len(imgs_repeated))
            try:
                normalized_features = np.concatenate((normalized_features, imgs_repeated))
            except ValueError:
                normalized_features = imgs_repeated

            labels = np.concatenate((labels, new_labels))
            assert len(normalized_features) == len(labels)

    preprocessed_label_stats = {ind: list(labels).count(ind) for ind in set(labels)}
    if len(normalized_features) > 0:
        features = np.concatenate((features, normalized_features))
    features = np.array([_convert_color_image(img) for img in features])

    return features, labels, orig_label_stats, preprocessed_label_stats


def _preprocess_train_test_data(train_features, train_labels, valid_features, valid_labels):
    """Todo: Initial steps towards some grayscaling etc. Remember, at this point the images have already been shuffled."""

    # TODO: find out ways to preprocess the data in meaningful ways.

    # halfway_train = len(X_train) // 2
    # train_1st_half_copy = X_train[:halfway_train]
    # train_2st_half_copy = X_train[halfway_train:]

    # labels_train_1st_half = y_train[:halfway_train]
    # labels_train_2st_half = y_train[halfway_train:]

    # Find the number of occurrences of each of the labels in the training data:

    orig_label_stats_train = {ind: list(train_labels).count(ind) for ind in set(train_labels)}
    orig_label_stats_valid = {ind: list(valid_labels).count(ind) for ind in set(valid_labels)}
    max_num_labels = max(orig_label_stats_train.values())

    for label, num in orig_label_stats_train.items():
        # Find all images with this label
        imgs = [img_label[0] for img_label in zip(train_features, train_labels) if img_label[1] == label]

        coeff = max_num_labels / orig_label_stats_train[label] - 1
        coeff_int = int(np.floor(coeff))
        original_length = len(imgs)
        if coeff_int > 0:
            imgs_repeated = np.array([img for img in imgs for _ in range(coeff_int)])
        else:
            imgs_repeated = np.array([])
        max_ind = int((coeff - coeff_int) * original_length)
        if max_ind > 0:
            if len(imgs_repeated) > 0:
                imgs_repeated = np.concatenate((imgs_repeated, imgs[:max_ind]))
            else:
                imgs_repeated = np.array(imgs[:max_ind])

        rotation_angles = [24 * (random() - 0.5) for _ in range(len(imgs_repeated))]
        scale_coeffs = [1 + 0.5 * (random() - 0.5) for _ in range(len(imgs_repeated))]

        for ind, img in enumerate(imgs_repeated):
            matr = cv2.getRotationMatrix2D((16, 16), rotation_angles[ind], scale_coeffs[ind])
            imgs_repeated[ind] = cv2.warpAffine(img, matr, (32, 32))

        if len(imgs_repeated) > 0:
            train_features = np.concatenate((train_features, imgs_repeated))
            train_labels = np.concatenate([train_labels, np.tile(label, len(imgs_repeated))])

    # Add a slightly scaled version of the other half of the images, with a little bit of added noise.

    normalized_train = [_convert_color_image(img) for img in train_features]
    normalized_valid = [_convert_color_image(img) for img in valid_features]

    preprocessed_label_stats_train = {ind: list(train_labels).count(ind) for ind in set(train_labels)}
    preprocessed_label_stats_valid = {ind: list(valid_labels).count(ind) for ind in set(valid_labels)}
    return normalized_train, train_labels, normalized_valid, valid_labels, orig_label_stats_train, \
           preprocessed_label_stats_train, orig_label_stats_valid, preprocessed_label_stats_valid


def normalize_luminosity_with_thre(img):
    """Normalize contrast as per http://stackoverflow.com/a/38312281 """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    # ychan = cv2.equalizeHist(img_yuv[:, :, 0])
    thre_ychan = cv2.adaptiveThreshold(img_yuv[:, :, 0] ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
    img_yuv[:, :, 0] = thre_ychan
    # convert the YUV image back to RGB format
    color_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return color_img


def _convert_color_image(img):
    return normalize_luminosity_with_thre(img)
    """Normalize contrast as per http://stackoverflow.com/a/38312281 """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    channel = img_yuv[:, :, 0]
    # plt.imshow(channel)
    # plt.show()
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    color_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return color_img


def _evaluate(X_data, y_data, batch_size, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def _LeNet(x, network):
    mu = 0.0
    sigma = 0.05

    F_W_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name=FIRST_CONVO_WEIGHTS_NAME)
    F_b_1 = tf.Variable(tf.zeros(6), name='first_convo_biases')
    strides_1 = [1, 1, 1, 1]
    padding_1 = 'VALID'
    conv1 = tf.nn.conv2d(x, F_W_1, strides=strides_1, padding=padding_1) + F_b_1

    conv1 = tf.nn.relu(conv1)

    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    pool1 = tf.nn.max_pool(conv1, ksize, strides, padding)

    F_W_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name=SECOND_CONVO_WEIGHTS_NAME)
    F_b_2 = tf.Variable(tf.zeros([16]), name='second_convo_biases')
    strides_2 = [1, 1, 1, 1]
    padding_2 = 'VALID'
    conv2 = tf.nn.conv2d(pool1, F_W_2, strides_2, padding_2) + F_b_2

    conv2 = tf.nn.relu(conv2)

    pool2 = tf.nn.max_pool(conv2, ksize, strides, padding)

    flat = flatten(pool2)
    flat = tf.nn.dropout(flat, 0.8)

    F_W_full_1 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name='first_full_weights')
    F_b_full_1 = tf.Variable(tf.zeros([120]), name='first_full_biases')
    full1 = tf.matmul(flat, F_W_full_1) + F_b_full_1
    full1 = tf.nn.relu(full1)

    F_W_full_2 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name='second_full_weights')
    F_b_full_2 = tf.Variable(tf.zeros([84]), name='second_full_biases')
    full2 = tf.matmul(full1, F_W_full_2) + F_b_full_2
    full2 = tf.nn.relu(full2)

    F_W_full_3 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma), name='third_full_weights')
    F_b_full_3 = tf.Variable(tf.zeros([43]), name='third_full_biases')
    full3 = tf.matmul(full2, F_W_full_3) + F_b_full_3

    return full3, network


def _define_model_architecture():
    """Define all the necessary tensorflow stuff here. Variables, losses, layer structure etc. ...

    TODO: Start with e.g. LeNet architecture, then figure out if something fancier should be tried out.

    :return: Nothing, just sets various network topology related tensors."""

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, None)
    network_topology = dict(x=x, y=y)

    one_hot_y = tf.one_hot(y, 43)

    logits, network_topology = _LeNet(x, network_topology)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

    network_topology['logits'] = logits
    network_topology['softmaxes'] = tf.nn.softmax(logits)
    network_topology['epochs'] = EPOCHS
    network_topology['batch_size'] = BATCH_SIZE

    loss_operation = tf.reduce_mean(cross_entropy)
    # step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE * 2, step, 100, 0.995)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    network_topology['loss_operation'] = loss_operation
    # network_topology['training_operation'] = optimizer.minimize(loss_operation, global_step=step)
    network_topology['training_operation'] = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    network_topology['accuracy_operation'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return network_topology


def _train_network_and_save_params(network, train_features, train_labels, valid_features, valid_labels, own_features, own_labels):
    """DRAFT: Train the network and print statistics. Also saves the model."""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(train_features)

        # Assigning to local variables for easier lookup
        batch_size = network['batch_size']
        epochs = network['epochs']
        training_operation = network['training_operation']
        accuracy_operation = network['accuracy_operation']
        x = network['x']
        y = network['y']

        print("Training...")
        print()
        for i in range(epochs):
            train_features, train_labels = shuffle(train_features, train_labels)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = train_features[offset:end], train_labels[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = _evaluate(valid_features, valid_labels, batch_size, accuracy_operation, x, y)
            print("{}: EPOCH {} ...".format(datetime.now(), i + 1))
            print("Validation Accuracy = {:.6f}".format(validation_accuracy))

            validation_accuracy_own_data = _evaluate(own_features, own_labels, batch_size, accuracy_operation, x, y)
            print("Validation Accuracy using own data = {:.6f}".format(validation_accuracy_own_data))

        name = tf.train.Saver().save(sess, './lenet-2.ckpt')
        print("Model saved, model name: {}".format(name))


def load_and_evaluate_model(features, labels, data_set_type="test"):
    """DRAFT: Load a model into a session again."""
    # some_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 6], 0, 0.001))
    network = _define_model_architecture()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        restorer.restore(sess, './lenet.ckpt')
        test_accuracy = _evaluate(features, labels, network['batch_size'], network['accuracy_operation'], network['x'], network['y'])
        for i in range(10):
            _print_max_softmaxes(sess, network, features[i], labels[i])
        print("{} set accuracy = {:.3f}".format(data_set_type, test_accuracy))

        # F_W_1 = next(var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if FIRST_CONVO_WEIGHTS_PATTERN.match(var.name))

        _print_featuremap(network, features[0])


def _print_max_softmaxes(sess, network, img, label):
    logits = sess.run(network['logits'], feed_dict={network['x']: np.array([img]), network['y']: np.array([label])})
    print("logits:")
    print(logits)
    softmaxes = sess.run(network['softmaxes'], feed_dict={network['x']: np.array([img]), network['y']: np.array([label])})
    print("softmaxes")
    print(softmaxes)
    max_softmaxes = tf.nn.top_k(softmaxes, k=5)
    print("max_softmaxes:")
    print("Real label: {}".format(label))
    print(sess.run(max_softmaxes))


def _print_featuremap(network, img):
    _outputFeatureMap(network, img, (network['convo_1'], network['pool_1'], network['convo_2'],
                                     network['pool_2']))

# TODO: the rest is just copy-paste from the initial workbook

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def _outputFeatureMap(network, image_input, tf_activations, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # activation = sess.run(tf_activation, feed_dict={network['x']: [image_input]})
        for tf_activation in tf_activations:
            activation = tf_activation.eval(session=sess, feed_dict={network['x']: [image_input]})
            try:
                featuremaps = activation.shape[3]
            except IndexError:
                featuremaps = 1
            plt.figure(figsize=(15,15))
            for featuremap in range(featuremaps):
                plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
                plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
                if activation_min != -1 & activation_max != -1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
                elif activation_max != -1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
                elif activation_min !=-1:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
                else:
                    plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            plt.show()


def load_and_preprocess_web_images(visualize=False):
    features, labels = _load_web_images_first_time()
    assert len(features) == len(labels)
    # Visualize first without preprocessing
    if visualize:
        _visualize_data(features, labels)
    # Resize
    features = np.array(list(map(lambda x: cv2.resize(x, (32, 32)), features)))
    if visualize:
        _visualize_data(features, labels)

    original_label_statas = {ind: list(labels).count(ind) for ind in set(labels)}
    features, labels, orig_label_stats, processed_label_stats = _preprocess_data(features, labels)
    assert len(features) == len(labels)
    preprocessed_label_stats = {ind: list(labels).count(ind) for ind in set(labels)}
    if visualize:
        _visualize_data(features, labels)

    return features, labels, original_label_statas, preprocessed_label_stats


def load_and_evaluate_own_images(visualize=False):
    features, labels, orig_label_stats, preprocessed_label_stats = load_and_preprocess_web_images(visualize=visualize)
    load_and_evaluate_model(features, labels, "own images")


if __name__ == "__main__":

    training_pipeline(visualize=False)
    # testing_pipeline()
    # testing_pipeline_web(visualize=False)
