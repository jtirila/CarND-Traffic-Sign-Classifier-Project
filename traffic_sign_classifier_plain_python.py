# Load pickled data
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import cv2

import random

# For testing purposes:
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# For file reading

# This is currently just a modification of the LeNet lab, architected for some modularity and better persistence etc.

# Some global variables

TRAINING_FILE = '/home/jtirila/Data/german-traffic-signs/train.p'
VALIDATION_FILE = '/home/jtirila/Data/german-traffic-signs/valid.p'
TESTING_FILE = '/home/jtirila/Data/german-traffic-signs/test.p'


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
    # TODO: this is to mitigate the current performance issues, remove when the data structure handling is better.
    # X_train = X_train[:1000]
    # X_valid = X_valid[:1000]
    # y_train = y_train[:1000]
    # y_valid = y_valid[:1000]

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
    for _ in range(3):
        index = random.randint(0, len(X_train))
        image = X_train[index].squeeze()

        plt.figure(figsize=(1, 1))
        plt.imshow(image, cmap='gray')
        plt.show()

        print(y_train[index])


def _preprocess_data(X_train, X_valid):
    """Todo: Initial steps towards some grayscaling etc."""

    # TODO: find out ways to preprocess the data in meaningful ways.
    return X_train, X_valid


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

def _first_convolutional_layer(input, mu, sigma):
    F_W = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mu, sigma), name='first_convo_weights')
    F_b = tf.Variable(tf.zeros([6]), name='first_convo_biases')

    strides = [1, 1, 1, 1]
    padding = 'VALID'

    return tf.add(tf.nn.conv2d(input, F_W, strides, padding), F_b)


def _second_convolutional_layer(input, mu, sigma):
    F_W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma), name='second_convo_weights')
    F_b = tf.Variable(tf.zeros([16]), name='second_convo_biases')
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    return tf.add(tf.nn.conv2d(input, F_W, strides, padding), F_b)


def _first_pooling(input):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    return tf.nn.max_pool(input, ksize, strides, padding)


def _second_pooling(input):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    return tf.nn.max_pool(input, ksize, strides, padding)


def _first_fully_connected(input):
    F_W = tf.Variable(tf.truncated_normal([400, 120]), name='first_full_weights')
    F_b = tf.Variable(tf.zeros([120]), name='first_full_biases')
    return tf.add(tf.matmul(input, F_W), F_b)


def _second_fully_connected(input):
    F_W = tf.Variable(tf.truncated_normal([120, 10]), name='second_full_weights')
    F_b = tf.Variable(tf.zeros([10]), name='second_full_biases')
    return tf.add(tf.matmul(input, F_W), F_b)



def _LeNet(x):

    mu = 0.0
    sigma = 0.1
    layer = _first_convolutional_layer(x, mu, sigma)
    layer = tf.nn.relu(layer)
    layer = _first_pooling(layer)
    layer = _second_convolutional_layer(layer, mu, sigma)
    layer = tf.nn.relu(layer)
    layer = _second_pooling(layer)
    layer = flatten(layer)
    layer = _first_fully_connected(layer)
    layer = tf.nn.relu(layer)
    layer = _second_fully_connected(layer)
    return layer


def _define_model_architecture():
    """Define all the necessary tensowflow stuff here. Variables, losses, layer structure etc. ...

    TODO: Start with e.g. LeNet architecture, then figure out if something fancier should be tried out.

    :return: Nothing, just sets various network topology related tensors."""

    x = tf.placeholder(tf.float32, (None, 32, 32, None))
    y = tf.placeholder(tf.int32, None)
    network_topology = dict(x=x)
    network_topology['y'] = y

    one_hot_y = tf.one_hot(y, 10)

    logits = _LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

    learning_rate = 0.001
    epochs = 500
    batch_size = 128

    network_topology['epochs'] = epochs
    network_topology['batch_size'] = batch_size

    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    network_topology['loss_operation'] = loss_operation
    network_topology['training_operation'] = optimizer.minimize(loss_operation)
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
    training_pipeline(mnist_test=True)
    # testing_pipeline()
