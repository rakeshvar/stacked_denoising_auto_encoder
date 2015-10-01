import os
import gzip
import pickle
import theano as th
import numpy as np


def share(data, dtype):
    return th.shared(np.asarray(data, dtype), borrow=True)


def share_xy(data_xy):
    return share(data_xy[0], th.config.floatX), share(data_xy[1], 'int32')


def load_mnist(dataset = 'data/mnist.pkl.gz'):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    data_dir, data_file = os.path.split(dataset)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.isfile(dataset):
        import urllib.request as url
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from ', origin)
        url.urlretrieve(origin, dataset)

    print('Loading data')
    f = gzip.open(dataset, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()

    return train_set, valid_set, test_set


def train_x_mnist(dataset = 'data/mnist.pkl.gz'):
    (rx, ry), (vx, vy), (sx, sy) = load_mnist()
    print("Input: ", rx.min(), rx.mean(), rx.max())
    return share(rx, th.config.floatX)

if __name__ == '__main__':
    (rx, ry), (vx, vy), (sx, sy) = load_mnist()
    for arr in (rx, ry, vx, vy, sx, sy):
        print(arr.shape)