from math import floor
import numpy as np



def spatialPool(data, dim, fn=np.max):
    height = data.shape[0]
    width = data.shape[1]

    wintv = width/(dim + 0.0)
    hintv = height/(dim + 0.0)

    out = np.ndarray(shape=(dim*dim), dtype=np.int32)
    idx = 0

    for i in range(dim):
        for j in range(dim):
            rmin = int(i*wintv)
            rmax = int((i+1)*wintv)
            cmin = int(j*hintv)
            cmax = int((j+1)*hintv)

            out[idx] = fn(data[rmin:rmax,cmin:cmax])
            idx += 1

    return out


def batchSpatialPool(data, dim, fn=np.max):
    out = np.ndarray([data.shape[0], np.sum(dim*dim), data.shape[3]], dtype=np.float16)

    for i in range(data.shape[0]):
        for j in range(data.shape[3]):
            idx = 0
            for k in dim:
                out[i, idx:idx+(k*k), j] = spatialPool(data[i, :, :, j], k)
                idx += k

    return out






data = np.ndarray([1, 4, 4, 256]);
dim = np.array([1,2])
print batchSpatialPool(data, dim).shape
