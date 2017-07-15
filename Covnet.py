from six.moves import cPickle as pickle
from scipy.misc import toimage
from random import shuffle
from PIL import Image

import tensorflow as tf
import numpy as np
import math
import sys
import os

dataRoot = './VOC2007'

boxPriors = [100.0, 100.0, 200.0, 100.0, 100.0, 200.0, 400.0, 200.0, 200.0, 400.0]
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
        union = tf.add(tf.abs(tf.multiply(tf.subtract(prop[3], prop[1], tf.subtract(prop[2], prop[0])))),
                       tf.abs(tf.multiply(tf.subtract(truth[3], truth[1]), tf.subtract(truth[2], truth[0]))))

        res = tf.cond(tf.equals(union, 0), 0, tf.truediv(intersection/union))

        def iou_f1(): return 0
        def iou_f2(): return res

        res = tf.cond(tf.logical_or(tf.greater(x1, x2), tf.greater(y1, y2)), iou_f1, iou_f2)
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
    tfIteration = tf.placeholder(tf.int32)

    numIter = 0
    for i in range(100):
        if(i == tfIteration):
            numIter = i
            break

    tf.print(tfIteration, [('Starting graph iteration %d' % numIter)])

    with open(os.path.join(dataRoot, 'batch', str(numIter) + '.pickle'), 'rb') as f:
        save = pickle.load(f)
        boxes = save['boxes']

    allTruth = boxes

    l1_weights = tf.Variable(tf.truncated_normal([7, 7, 3, 96], stddev=0.1))
    l1_biases = tf.Variable(tf.zeros([96]))

    l2_weights = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.1))
    l2_biases = tf.Variable(tf.zeros([256]))

    l3_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.1))
    l3_biases = tf.Variable(tf.zeros([384]))

    l4_weights = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.1))
    l4_biases = tf.Variable(tf.zeros([384]))

    l5_weights = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
    l5_biases = tf.Variable(tf.zeros([256]))

    fcn_weights = tf.Variable(tf.truncated_normal([1, 1, 256, 5*numProps], stddev=0.1))
    fcn_biases = tf.Variable(tf.zeros(5*numProps))

    def compute(tfTrainDataset):
        conv = tf.nn.conv2d(tfTrainDataset, l1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l1_biases)
        pool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1,2,2,1], 'SAME')

        conv = tf.nn.conv2d(pool, l2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l2_biases)
        pool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1,2,2,1], 'SAME')

        conv = tf.nn.conv2d(pool, l3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l3_biases)

        conv = tf.nn.conv2d(hidden, l4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + l4_biases)

        conv = tf.nn.conv2d(hidden, l5_weights, [1, 3, 3, 1], padding='VALID')
        hidden = tf.nn.relu(conv + l5_biases)

        fcn = tf.nn.conv2d(hidden, fcn_weights, [1, 1, 1, 1], padding='SAME')
        props = tf.nn.relu(fcn + fcn_biases)

        return props

    def loss(allProps, allTruth):
        loss = 0.0


        def batchIter(b):
            props = allProps[b]
            truth = allTruth[b]


            def rowIter(i):

                def colIter(i, j):
                    
                    def hasBox(i, j):
                        box = -1
                        for idx in range(len(truth)):
                            tx, ty, _, _ = format(truth[idx])

                            def hasbox_f1(): return idx
                            def haxbox_f2(): return box

                            box = tf.cond(tf.logical_and(tf.equals(i, tf.floordiv(tx, 50)), tf.equals(j, tf.floordiv(ty, 50))), hasBox_f1, hasBox_f2)
                        return box








        for b in range(batchSize):

            print('processing props for batch')

            props = allProps[b]
            truth = allTruth[b]

            for i in range(props.shape[0]):
                for j in range(props.shape[1]):

                    m = -1
                    for idx in range(len(truth)):
                        tx, ty, tw, th = format(truth[idx])
                        ix = tx/50
                        jx = ty/50
                        if(ix == i and jx == j):
                            m = idx
                            print(ix, jx, m)

                    if m == -1:
                        for k in range(numProps):
                            loss = tf.add(loss, tf.multiply(-1.0, tf.log(tf.subtract(1.0, tf.sigmoid(props[i, j, k+4])))))
                    else:
                        box = truth[m]

                        maxIOU = 0
                        maxProb = 0
                        maxProp = []

                        for k in range(numProps):
                            px = tf.floor(tf.multiply(50.0, tf.add(i, tf.sigmoid(props[i, j, 5*k]))))
                            py = tf.floor(tf.multiply(50.0, tf.add(i, tf.sigmoid(props[i, j, 5*k+1]))))
                            pw = tf.floor(tf.multiply(boxPriors[2*k], tf.exp(props[i, j, 5*k+2])))
                            ph = tf.floor(tf.multiply(boxPriors[2*k+1], tf.exp(props[i, j, 5*k+3])))
                            prop = [px, py, pw, ph]
                            if(iou(coords(prop), box) > maxIOU):
                                maxIOU = iou(prop, box)
                                maxProb = tf.sigmoid(props[i, j, 5*k+4])
                                maxProp = prop

                        [px, py, pw, ph] = maxProp

                        loss = tf.add(loss, tf.square(tf.subtract(tx, px)),
                                            tf.square(tf.subtract(ty, py)),
                                            tf.square(tf.subtact(tf.sqrt(tw), tf.sqrt(pw))),
                                            tf.square(tf.subtact(tf.sqrt(th), tf.sqrt(ph))))

                        loss = tf.add(loss, tf.multiply(-1, tf.log(tf.sigmoid(maxProb))))
        return loss

    print('Generating props')
    allProps = compute(tfTrainDataset)

    print('Calculating loss')
    loss = loss(allProps, allTruth)
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


    if(numIter%10 == 0 and numIter > 0):
        allProps = compute(tfTestDataset)
        drawData = tfTestDataset

        for b in range(batchSize):
            props = allProps[b]

            for i in range(props.shape[0]):
                for j in range(props.shape[1]):
                    maxProp = 0;
                    maxProb = 0;
                    for k in range(numProps):
                        px = tf.floor(tf.multiply(50, tf.add(i, tf.sigmoid(props[i, j, 5*k]))))
                        py = tf.floor(tf.multiply(50, tf.add(i, tf.sigmoid(props[i, j, 5*k+1]))))
                        pw = tf.floor(tf.multiply(boxPriors[2*k], tf.exp(props[i, j, 5*k+2])))
                        ph = tf.floor(tf.multiply(boxPriors[2*k+1], tf.exp(props[i, j, 5*k+3])))
                        prop = [px, py, pw, ph]

                        prob = tf.sigmoid(props[i, j, 5*k+4])

                        if(prob > maxProb):
                            maxProp = prop
                            maxProb = prob

                    if(maxProb > 0.7):
                        drawData[b, :, :, :] = drawRect(drawData[b], coords(maxProp))

        try:
            with open(os.path.join(dataRoot, 'test', 'predictions.pickle'), 'wb') as f:
                pickle.dump(drawData, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('unable to save predictions %s' % e)


num_steps = 90

with tf.Session(graph=graph) as session:
    print('hi there')
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()


    for step in range(num_steps):

        print('hello')
        print(os.path.join(dataRoot, 'batch', (str(step) + '.pickle')))

        with open(os.path.join(dataRoot, 'batch', (str(step) + '.pickle')), 'rb') as f:
            save = pickle.load(f)
            images = save['images']
            boxes = save['boxes']

        feed_dict = {tfTrainDataset : images, tfIteration: step, tfTestDataset : testDataset}
        _, l  = session.run([optimizer, loss], feed_dict=feed_dict)

        print('Minibatch training loss %f' % l)

        if(step%30 == 0 and step != 0):
            savePath = saver.save(session, os.path.join(dataRoot, 'models', (str(step) + '_mdl.ckpt')))
            print('Model saved to file: %s' % savePath)

        print('About to iterate')
