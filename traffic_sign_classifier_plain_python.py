# Load pickled data
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

# These are for image transformations
import cv2
import csv

import random

# For testing purposes:
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Some global variables

TRAINING_FILE = '/home/jtirila/Data/german-traffic-signs/train.p'
VALIDATION_FILE = '/home/jtirila/Data/german-traffic-signs/valid.p'
TESTING_FILE = '/home/jtirila/Data/german-traffic-signs/test.p'
LABEL_FILE = 'signnames.csv'


LEARNING_RATE = 0.0014
EPOCHS = 100
BATCH_SIZE = 128


def testing_pipeline():
    """Load test data and previously saved model, print statistics."""
    X_test, y_test = _load_real_validation_data()
    load_and_evaluate_model(X_test, y_test)


def training_pipeline(mnist_test=True):
    """Load training data, define and train the model, then save to disk. """
    if mnist_test:
        X_train, y_train, X_valid, y_valid = _load_test_data()
    else:
        train, valid, test = _load_previously_saved_data()
        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_train, y_train = shuffle(X_train, y_train)
        X_valid, y_valid = shuffle(X_valid, y_valid)

    # _print_training_data_basic_summary()
    # X_train, y_train, X_valid, y_valid = X_train[:10000], y_train[:10000], X_valid[:2000], y_valid[:2000]

    _print_training_data_basic_summary(X_train, y_train, X_valid, y_valid)

    if not mnist_test:
        X_train, X_valid = _preprocess_data(X_train, X_valid)
    _visualize_data(X_train, y_train)
    # print("Moving on")

    # Work with the actual model begins
    network = _define_model_architecture()
    _train_network_and_save_params(network, X_train, y_train, X_valid, y_valid)


def _load_real_validation_data():
    train, valid, test = _load_previously_saved_data()
    X_test, y_test = test['features'], test['labels']
    X_test, y_test = shuffle(X_test, y_test)
    return X_test, y_test


def _load_test_validation_data():
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test, y_test = shuffle(X_test, y_test)
    return X_test, y_test



def _augment_image_data(image):
    """FIXME get image data as input, return same structure but an augmented version."""



def _load_test_data():
    """Loads the mnist character data. This is a short-term placeholder to get building something before getting the
    real data set prepared."""
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    # Pad images with 0s
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    # Shuffle the data
    X_train, y_train, = \
        shuffle(X_train, y_train)
    X_validation, y_validation = \
        shuffle(X_validation, y_validation)
    return X_train, y_train, X_validation, y_validation


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


def _print_test_data_basic_summary(X_test, y_test):
    """Todo: write this for test data set."""
    pass


def _print_training_data_basic_summary(X_train, y_train, X_valid, y_valid):
    # ### Replace each question mark with the appropriate value.
    # ### Use python, pandas or numpy methods rather than hard coding the results

    # return
    # # TODO: Number of training examples
    n_train = len(X_train)
    n_valid = len(X_valid)


    # # TODO: What's the shape of an traffic sign image?
    shape = X_train[0].shape
    image_shape = "{} x {} x {}".format(shape[0], shape[1], shape[2])

    n_classes = len(set(y_train))

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Image data shape = ", image_shape)
    print("Number of classes =", n_classes)


def _visualize_data(X_train, y_train):
    label_dict = {}
    with open('signnames.csv', 'r') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            label_dict[row['ClassId']] = row['SignName']

    for i in range(1, 17):
        plt.subplot(4,4,i)
        plt.imshow(X_train[i - 1])
        plt.axis('off')
        plt.title("{}: {}".format(y_train[i - 1], label_dict[str(y_train[i - 1])][:19]))
    plt.show()


def _preprocess_data(X_train, X_valid):
    """Todo: Initial steps towards some grayscaling etc."""

    # TODO: find out ways to preprocess the data in meaningful ways.
    normalized_train = []
    normalized_valid = []

    # http://stackoverflow.com/a/38312281


    for img in X_train:
        normalized_train.append(_convert_color_image(img))
    for img in X_valid:
        normalized_valid.append(_convert_color_image(img))

    return normalized_train, normalized_valid

def _convert_color_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def _evaluate(X_data, y_data, batch_size, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO: these layer methods probably don't make sense on their own, just include them all in the network architecture
# TODO: definition


def _first_convo(x, mu, sigma):
    F_W_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name='first_convo_weights')
    F_b_1 = tf.Variable(tf.zeros(6), name='first_convo_biases')
    strides_1 = [1, 1, 1, 1]
    padding_1 = 'VALID'
    conv1 = tf.nn.conv2d(x, F_W_1, strides=strides_1, padding=padding_1) + F_b_1
    return conv1


def _first_pooling(x):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    pool1 = tf.nn.max_pool(x, ksize, strides, padding)
    return pool1


def _second_convo(x, mu, sigma):
    F_W_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name='second_convo_weights')
    F_b_2 = tf.Variable(tf.zeros([16]), name='second_convo_biases')
    strides_2 = [1, 1, 1, 1]
    padding_2 = 'VALID'
    conv2 = tf.nn.conv2d(x, F_W_2, strides_2, padding_2) + F_b_2
    return conv2


def _second_pooling(x):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    pool2 = tf.nn.max_pool(x, ksize, strides, padding)
    return pool2


def _first_full(x, mu, sigma):
    F_W_full_1 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name='first_full_weights')
    F_b_full_2 = tf.Variable(tf.zeros([120]), name='first_full_biases')
    full1 = tf.matmul(x, F_W_full_1) + F_b_full_2
    full1 = tf.nn.relu(full1)
    return full1


def _second_full(x, mu, sigma):
    F_W_full_2 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name='second_full_weights')
    F_b_full_2 = tf.Variable(tf.zeros([84]), name='second_full_biases')
    full2 = tf.matmul(x, F_W_full_2) + F_b_full_2
    return tf.nn.relu(full2)


def _third_full(x, mu, sigma):
    F_W_full_3 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma), name='third_full_weights')
    F_b_full_3 = tf.Variable(tf.zeros([43]), name='third_full_biases')
    return tf.matmul(x, F_W_full_3) + F_b_full_3


def _LeNet(x):
    mu = 0.0
    sigma = 0.1

    conv1 = _first_convo(x, mu, sigma)
    conv1 = tf.nn.l2_normalize(conv1, 0)

    pool1 = _first_pooling(conv1)
    pool1 = tf.nn.l2_normalize(pool1, 0)

    conv2 = _second_convo(pool1, mu, sigma)
    conv2 = tf.nn.l2_normalize(conv2, 0)
    pool2 = _second_pooling(conv2)

    flat = flatten(pool2)
    flat = tf.nn.l2_normalize(flat, 0)

    full1 = _first_full(flat, mu, sigma)
    full1 = tf.nn.l2_normalize(full1, 0)
    full2 = _second_full(full1, mu, sigma)
    full2 = tf.nn.l2_normalize(full2, 0)

    full3 = _third_full(full2, mu, sigma)
    full3 = tf.nn.l2_normalize(full3, 0)

    return full3


def _define_model_architecture():
    """Define all the necessary tensorflow stuff here. Variables, losses, layer structure etc. ...

    TODO: Start with e.g. LeNet architecture, then figure out if something fancier should be tried out.

    :return: Nothing, just sets various network topology related tensors."""

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, None)
    network_topology = dict(x=x)
    network_topology['y'] = y

    one_hot_y = tf.one_hot(y, 43)

    logits = _LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

    network_topology['epochs'] = EPOCHS
    network_topology['batch_size'] = BATCH_SIZE

    loss_operation = tf.reduce_mean(cross_entropy)
    step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE * 2, step, 100, 0.995)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    network_topology['loss_operation'] = loss_operation
    network_topology['training_operation'] = optimizer.minimize(loss_operation, global_step=step)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    network_topology['accuracy_operation'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return network_topology


def _train_network_and_save_params(network, X_train, y_train, X_valid, y_valid):
    """DRAFT: Train the network and print statistics. Also saves the model."""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

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
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = _evaluate(X_valid, y_valid, batch_size, accuracy_operation, x, y)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.6f}".format(validation_accuracy))

        name = tf.train.Saver().save(sess, './lenet.ckpt')
        print("Model saved, model name: {}".format(name))


def load_and_evaluate_model(X_test, y_test):
    """DRAFT: Load a model into a session again."""
    # some_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 6], 0, 0.001))
    network = _define_model_architecture()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        restorer.restore(sess, './lenet.ckpt')
        test_accuracy = _evaluate(X_test, y_test, network['batch_size'], network['accuracy_operation'], network['x'], network['y'])
        print("Test Accuracy = {:.3f}".format(test_accuracy))


# TODO: the rest is just copy-paste from the initial workbook

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def _outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
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


if __name__ == "__main__":
    training_pipeline(mnist_test=False)
    # testing_pipeline()
