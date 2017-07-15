from six.moves import cPickle as pickle
from scipy.misc import toimage
from random import shuffle
from PIL import Image
import pdb

import tensorflow as tf
import numpy as np
import math
import sys
import os

dataRoot = './VOC2007'


xP = np.tile(np.arange(0, 500, 100), (5, 1))
yP = np.transpose(xP)
xPriors = np.transpose(np.tile(yP, (5, 1, 1)))
yPriors = np.transpose(np.tile(xP, (5, 1, 1)))

wPriors = [400, 200, 400, 500, 300]
hPriors = [400, 400, 200, 300, 500]
wPriors = np.tile(wPriors, (5, 5, 1))
hPriors = np.tile(hPriors, (5, 5, 1))

batchSize = 50;
numProps = 5;
spWin = 3

print('loading data')

with open(os.path.join(dataRoot, 'batch', '96.pickle'), 'rb') as f:
    save = pickle.load(f)
    testDataset = save['images']
    testLabels = save['boxes']

print(testDataset.shape, 'loaded test data')

graph = tf.Graph()
with graph.as_default():

    def checkForNan(tensor):
        return tf.reduce_sum(tf.add(tf.to_float(tf.is_nan(tensor)), tf.to_float(tf.is_inf(tensor))))

    def drawRect(image, prop):
        for i in range(prop[0], prop[2]):
            image[i, prop[1], :] = [255, 0, 0]
            image[i, prop[3], :] = [255, 0, 0]

        for j in range(prop[1], prop[3]):
            image[prop[0], j, :] = [255, 0, 0]
            image[prop[2], j, :] = [255, 0, 0]

        return image


    def iou(prop, truth):
        x1 = tf.maximum(prop[0], truth[0])
        y1 = tf.maximum(prop[1], truth[1])

        x2 = tf.minimum(prop[2], truth[2])
        y2 = tf.minimum(prop[3], truth[3])


        intersection = tf.abs(tf.multiply(tf.subtract(x1, x2), tf.subtract(y1, y2)))
        union = tf.add(tf.abs(tf.multiply(tf.subtract(prop[3], prop[1]), tf.subtract(prop[2], prop[0]))),
                       tf.abs(tf.multiply(tf.subtract(truth[3], truth[1]), tf.subtract(truth[2], truth[0]))))


        res = tf.truediv(intersection, union)
        res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
        return res


    def coords(prop):
        x1 = tf.subtract(prop[0], tf.div(prop[2], 2))
        y1 = tf.subtract(prop[1], tf.div(prop[3], 2))
        x2 = tf.add(prop[0], tf.div(prop[2], 2))
        y2 = tf.add(prop[1], tf.div(prop[3], 2))

        return x1, y1, x2, y2


    def format(truth):
        w = tf.subtract(truth[2], truth[0])
        h = tf.subtract(truth[3], truth[1])
        x = tf.add(truth[0], tf.div(w, 2))
        y = tf.add(truth[1], tf.div(h, 2))

        return x, y, w, h

    tfTrainDataset = tf.placeholder(tf.float32, shape=(batchSize, 500, 500, 3))
    tfTestDataset = tf.placeholder(tf.float32, shape=(batchSize, 500, 500, 3))
    tfTrainBoxes = tf.placeholder(tf.float32, shape=(batchSize, 5, 5, 5))
    tfIteration = tf.placeholder(tf.int32)

    numIter = 0
    for i in range(100):
        if(i == tfIteration):
            numIter = i
            break



    l1_weights = tf.Variable(tf.truncated_normal([7, 7, 3, 32], stddev=0.1, dtype=tf.float32))
    l1_biases = tf.Variable(tf.zeros([32]))

    l2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
    l2_biases = tf.Variable(tf.zeros([64]))

    l3_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=0.1, dtype=tf.float32))
    l3_biases = tf.Variable(tf.zeros([96]))

    l4_weights = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=0.1, dtype=tf.float32))
    l4_biases = tf.Variable(tf.zeros([96]))

    l5_weights = tf.Variable(tf.truncated_normal([3, 3, 96, 64], stddev=0.1, dtype=tf.float32))
    l5_biases = tf.Variable(tf.zeros([64]))

    fcn_weights = tf.Variable(tf.truncated_normal([1, 1, 64, 5*numProps], stddev=0.1, dtype=tf.float32))
    fcn_biases = tf.Variable(tf.zeros(5*numProps))

    def compute(tfTrainDataset):
        conv = tf.nn.conv2d(tfTrainDataset, l1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l1_biases)
        pool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1,2,2,1], 'SAME')

        AA = checkForNan(pool)

        conv = tf.nn.conv2d(pool, l2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l2_biases)
        pool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1,2,2,1], 'SAME')

        BB = checkForNan(pool)


        CC = checkForNan(l3_weights)

        conv = tf.nn.conv2d(pool, l3_weights, [1, 1, 1, 1], padding='SAME')

        hidden = tf.nn.relu(conv + l3_biases)


        conv = tf.nn.conv2d(hidden, l4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l4_biases)

        DD = checkForNan(hidden)

        conv = tf.nn.conv2d(hidden, l5_weights, [1, 3, 3, 1], padding='VALID')
        hidden = tf.nn.relu(conv + l5_biases)

        EE = checkForNan(hidden)

        fcn = tf.nn.conv2d(hidden, fcn_weights, [1, 2, 2, 1], padding='SAME')
        props = tf.nn.relu(fcn + fcn_biases)

        return props, AA, BB, CC, DD, EE



    def loss(props, boxes):

        xIdx = np.arange(0, 25, 5)
        yIdx = np.arange(1, 26, 5)
        wIdx = np.arange(2, 27, 5)
        hIdx = np.arange(3, 28, 5)
        cIdx = np.arange(4, 29, 5)

        def select(batch, idxs):
            return tf.stack([batch[:, :, :, i] for i in idxs], axis=3)

        nonRegionLoss = tf.multiply(-1.0, tf.log(tf.subtract(1.1, tf.sigmoid(select(props, cIdx)))))
        nonRegionLoss = tf.multiply(nonRegionLoss, tf.subtract(1.0, boxes[:, :, :, 1:2]))

        FF = nonRegionLoss
        # FF = tf.zeros(1)


        # px = tf.add(tf.multiply(100.0, tf.sigmoid(select(props, xIdx))), xPriors)
        # py = tf.add(tf.multiply(100.0, tf.sigmoid(select(props, yIdx))), yPriors)
        # pw = tf.multiply(tf.sigmoid(select(props, wIdx)), wPriors)
        # ph = tf.multiply(tf.sigmoid(select(props, hIdx)), hPriors)
        #
        # tx = select(boxes, [1])
        # ty = select(boxes, [2])
        # tw = select(boxes, [3])
        # th = select(boxes, [4])
        #
        # px1 = tf.maximum(0.0, tf.subtract(px, tf.divide(pw, 2)))
        # py1 = tf.maximum(0.0, tf.subtract(py, tf.divide(ph, 2)))
        # px2 = tf.maximum(500.0, tf.add(px, tf.divide(pw, 2)))
        # py2 = tf.maximum(500.0, tf.add(py, tf.divide(ph, 2)))
        #
        # tx1 = tf.subtract(tx, tf.divide(tw, 2))
        # ty1 = tf.subtract(ty, tf.divide(th, 2))
        # tx2 = tf.add(tx, tf.divide(tw, 2))
        # ty2 = tf.add(ty, tf.divide(th, 2))
        #
        # res = iou([px1, py1, px2, py2], [tx1, ty1, tx2, ty2])
        #
        # maxMask = tf.reduce_max(res, axis=3)
        # maxMask = tf.expand_dims(maxMask, 3)
        # maxMask = tf.to_float(tf.greater_equal(res, maxMask))
        #
        # regionLoss = tf.add_n([tf.sqrt(tf.square(tf.subtract(tx, px))),
        #                     tf.sqrt(tf.square(tf.subtract(ty, py))),
        #                     tf.sqrt(tf.square(tf.subtract(tf.sqrt(tw), tf.sqrt(pw)))),
        #                     tf.sqrt(tf.square(tf.subtract(tf.sqrt(th), tf.sqrt(ph))))])
        #
        # regionLoss = tf.multiply(regionLoss, maxMask)
        #
        # regionLoss = tf.multiply(regionLoss, boxes[:, :, :, 1:2])
        #
        # regionLoss = tf.add(regionLoss, tf.multiply(tf.multiply(-1.0, boxes[:, :, :, 1:2]), tf.log(tf.sigmoid(select(props, cIdx)))))
        #
        # GG = regionLos
        GG = tf.zeros(1)

        # loss = tf.add(regionLoss, nonRegionLoss)
        # loss = tf.reduce_sum(loss)
        loss = tf.reduce_sum(nonRegionLoss)
        return loss,FF, GG

    print('Generating props')
    props, AA, BB, CC, DD, EE = compute(tfTrainDataset)

    print('Calculating loss')
    loss, FF, GG = loss(props, tfTrainBoxes)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


    if(numIter%10 == 0 and numIter > 0):
        props = compute(tfTestDataset)
        preds = []

        def select(batch, idxs):
            return tf.stack([batch[:, :, :, i] for i in idxs], axis=3)

        xIdx = np.arange(0, 25, 5)
        yIdx = np.arange(1, 26, 5)
        wIdx = np.arange(2, 27, 5)
        hIdx = np.arange(3, 28, 5)
        cIdx = np.arange(4, 29, 5)

        px = tf.add(tf.multiply(100.0, tf.sigmoid(select(props, xIdx))), xPriors)
        py = tf.add(tf.multiply(100.0, tf.sigmoid(select(props, yIdx))), yPriors)
        pw = tf.multiply(tf.sigmoid(select(props, wIdx)), wPriors)
        ph = tf.multiply(tf.sigmoid(select(props, hIdx)), hPriors)

        px1 = tf.maximum(0.0, tf.subtract(px, tf.divide(pw, 2)))
        py1 = tf.maximum(0.0, tf.subtract(py, tf.divide(ph, 2)))
        px2 = tf.maximum(500.0, tf.add(px, tf.divide(pw, 2)))
        py2 = tf.maximum(500.0, tf.add(py, tf.divide(ph, 2)))

        cids = tf.sigmoid(select(props, cIdx))
        maxMask = tf.reduce_max(cids, axis=3)
        maxMask = tf.expand_dims(maxMask, 3)

        cids = tf.to_float(tf.greater_equal(cids, maxMask))
        cids = tf.to_float(tf.greater_equal(cids, 0.7))

        for b in range(cids.shape[0]):
            batchRects = []
            for i in range(cids.shape[1]):
                for j in range(cids.shape[2]):
                    for k in range(cids.shape[3]):
                        if(cids[b, i, j, k] == 1):
                            rect = [px1[b, i, j, k],
                                    py1[b, i, j, k],
                                    px2[b, i, j, k],
                                    py2[b, i, j, k]]
                            batchRects.append(rect)
            preds.append(batchRects)

        try:
            with open(os.path.join(dataRoot, 'test', 'predictions.pickle'), 'wb') as f:
                pickle.dump(preds, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('unable to save predictions %s' % e)

num_steps = 90

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    for step in range(num_steps):
        print(os.path.join(dataRoot, 'batch', (str(step) + '.pickle')))

        with open(os.path.join(dataRoot, 'batch', (str(step) + '.pickle')), 'rb') as f:
            save = pickle.load(f)
            images = save['images']
            boxes = save['boxes']

        feed_dict = {tfTrainDataset : images, tfTrainBoxes : boxes, tfIteration: step, tfTestDataset : testDataset}
        _, l, AA1, BB1, CC1, DD1, EE1, FF1, GG1  = session.run([optimizer, loss, AA, BB, CC, DD, EE, FF, GG], feed_dict=feed_dict)

        pdb.set_trace()

        print('Minibatch training loss %10.2f' % l)

        if(step%30 == 0 and step != 0):
            savePath = saver.save(session, os.path.join(dataRoot, 'models', (str(step) + '_mdl.ckpt')))
            print('Model saved to file: %s' % savePath)

        print('About to iterate')
