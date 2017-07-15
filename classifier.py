from six.moves import cPickle as pickle
from six.moves import range
from scipy import ndimage

import tensorflow as tf
import numpy as np
import fnmatch
import sys
import os

data_root = './data'
img_h = 604
img_w = 400
img_d = 3

def store_as_pickle():
    prefixes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    files = os.listdir(data_root)

    dataset = np.ndarray(shape=(len(files), img_h, img_w, img_d), dtype=np.float32)
    labels = np.ndarray(shape=(len(files)), dtype=np.int32)

    idx = 0
    grp = 0
    for pref in prefixes:
        imgs = fnmatch.filter(files, pref + '*')
        for img in imgs:
            img_data = ndimage.imread(os.path.join(data_root, img)).astype(float)
            dataset[idx, :, :, :] = img_data
            labels[idx] = grp
            idx += 1
        grp += 1

    print('Full dataset tensor:', dataset.shape)
    print('Max:', np.max(dataset))

    try:
        with open('./dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data', e)

    try:
        with open('./labels.pickle', 'wb') as f:
            pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data', e)

#store_as_pickle()


def train_model():
    with open('./dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
    with open('./labels.pickle', 'rb') as f:
        labels = pickle.load(f)

    dataset = dataset.reshape((-1, img_h*img_w*img_d))

    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm, :]
    labels = labels[perm]

    num_labels = 22
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.int32)

    train_size = 500
    test_size = dataset.shape[0] - train_size

    train_dataset = np.ndarray((train_size, img_h*img_w*img_d), dtype=np.float32)
    test_datset = np.ndarray((test_size, img_h*img_w*img_d), dtype=np.float32)

    train_labels = np.ndarray((train_size, num_labels))
    test_labels = np.ndarray((test_size, num_labels))

    train_dataset = dataset[:train_size, :]
    test_dataset = dataset[(train_size + 1):, :]
    train_labels = labels[:train_size, :]
    test_labels = labels[(train_size + 1):, :]

    graph = tf.Graph()
    with graph.as_default():

        tf_train_dataset = tf.constant(train_dataset)
        tf_train_labels = tf.constant(train_labels)

        tf_test_dataset = tf.constant(test_dataset)
        tf_test_labels = tf.constant(test_labels)

        weights = tf.Variable(
            tf.truncated_normal([img_h * img_w * img_d, num_labels]))

        biases = tf.Variable(tf.zeros([num_labels]))


        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    def accuracy(predictions, labels):
        return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    num_steps = 200

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        for step in range(num_steps):

            feed_dict = {tf_train_dataset : train_dataset, tf_train_labels : train_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 50 == 0):
                print('Training acc: %.1f%%' % accuracy(predictions, train_labels))
                print('Testing acc: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

train_model()
