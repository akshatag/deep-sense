from six.moves import cPickle as pickle
from matplotlib import pyplot as plt
from scipy.misc import toimage
from scipy import ndimage

import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import sys
import os

# Global scope
dataRoot = './VOC2007'
batchSize = 50


def loadBoxes():
    fdir = os.path.join(dataRoot, 'Annotations')
    files = os.listdir(fdir)

    boxes = np.zeros((len(files), 5, 5, 5), dtype=np.float16)
    idx = 0

    for f in files:
        tree = ET.parse(os.path.join(fdir, f))
        root = tree.getroot()

        for obj in root.iter('bndbox'):
            x1 = float(obj[0].text)
            y1 = float(obj[1].text)
            x2 = float(obj[2].text)
            y2 = float(obj[3].text)

            w = x2 - x1
            h = y2 - y1
            x = x1 + w/2
            y = y1 + h/2

            i = int(x/100)
            j = int(y/100)

            boxes[idx, i, j, :] = np.array([1, x, y, w, h])

        idx += 1

    return boxes


def loadImages():
    fdir = os.path.join(dataRoot, 'JPEGImages')
    files = os.listdir(fdir)

    images = np.zeros((len(files), 500, 500, 3), dtype=np.float16)
    idx = 0

    for f in files:
        imgData = ndimage.imread(os.path.join(fdir, f)).astype(float)

        xlen = min(500, imgData.shape[0])
        ylen = min(500, imgData.shape[1])

        images[idx, 0:xlen, 0:ylen, :] = imgData
        idx += 1

    print images.shape
    return images


def pickleData():
    boxes = loadBoxes()
    images = loadImages()

    perm = np.random.permutation(images.shape[0])
    images = images[perm, :, :, :]
    boxes = boxes[perm, :, :, :]

    for i in range(0, images.shape[0]/batchSize):
        imgBatch = images[batchSize*i:batchSize*(i+1), :, :, :]
        boxBatch = boxes[batchSize*i:batchSize*(i+1)]

        save = {
            'images' : imgBatch,
            'boxes' : boxBatch
        }

        try:
            with open(os.path.join(dataRoot, 'batch', (str(i) + '.pickle')), 'wb') as f:
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            print('Saved %d' % i)
        except Exception as e:
            print('unable to save batch %d' % i, e)

pickleData()
