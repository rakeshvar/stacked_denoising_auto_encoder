#!/usr/bin/python
import pickle
import numpy as np
import theano
import theano.tensor as tt
import PIL.Image
import numpy.linalg as la
import dae

from utils import tile_raster_images
from weights import gen_random_gabor_wts

NORM_THRESHOLD = .05
batch_size = 20
floatX = theano.config.floatX


def get(self, borrow=True):
    return self.get_value(borrow=borrow)

theano.tensor.sharedvar.SharedVariable.get = get


def get_shape(x):
    return x.get().shape


def minmeanmax(x):
    return np.array((x.min(), x.mean(), x.max()))


def print_zmmm(arr, head):
    print('{2}\tZeros:{0:3d}\tMin:{1[0]:7.3f}'
          '\tMean:{1[1]:7.3f}\tMax:{1[2]:7.3f}'
          ''.format(sum(arr < NORM_THRESHOLD ) if arr.ndim else 0,
                    minmeanmax(arr), head))


def print_bias_etc(da, file_name, epch, side, save_raster=False, freq=10):
    # Save raster image once in a while and print avg. bias too
    # Find the two norm of each loading
    norms = np.apply_along_axis(la.norm, 0, da.w.get())
    print_zmmm(norms, "  Weight norms")
    print_zmmm(da.b.get(), "  Biases ")
    print_zmmm(da.b_prime.get(), "  Biases'")

    if save_raster:
        if epch % freq == 0:
            print_tile(da.w.get(False).T, file_name.format(epch), (side, side))

        with open(file_name[:-8] + '.csv', 'ab') as f:
            np.savetxt(f, norms, newline=',', fmt='%.4f')
            f.write(b'\n')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Stack_DAE(object):
    def __init__(self, x, n_nodes, lambda1, lr, noise, n_epochs):
        assert (len(lambda1) == len(lr) == len(noise) == len(n_nodes) == 3)

        self.x1 = x
        self.n_batches, n_visible = get_shape(x)
        self.n_batches //= batch_size
        self.n_nodes = n_nodes
        self.lambda1 = lambda1
        self.lr = lr
        self.noise = noise
        self.n_epochs = n_epochs

        side = int(n_visible ** .5)
        assert (side ** 2 == n_visible)  # Need perfect square
        self.side = side

        print("Initializing Weights for First Layer")
        w0 = np.empty((n_visible, n_nodes[0]))
        for i in range(w0.shape[1]):
            w0[:, i] = gen_random_gabor_wts((side, side)).flatten()

        self.first, self.first_train = \
            self.init_next(inpt=x, w=w0, index=0, binary_data=False)

    def init_next(self, inpt, index, w=None, binary_data=False):
        print("\nAdding DAE Layer")
        n_in = get_shape(inpt)[1]

        my_dae = dae.DAE(w=w, n_visible=n_in, n_hidden=self.n_nodes[index],
                     lambda1=self.lambda1[index])

        costs, updates = my_dae.get_cost_updates(noise=self.noise[index],
                                                 binary_data=binary_data,
                                                 init_learning_rate=self.lr[index])

        index = tt.lscalar()

        train_dae = theano.function(
            [index],
            costs,
            updates=updates,
            givens={my_dae.x: inpt[index*batch_size:(index + 1)*batch_size]})

        print(my_dae)
        return my_dae, train_dae

    def train_this(self, da, train_da, n_epochs, file_name='',
                   save_raster=False):
        print("\nTraining DAE Layer")
        prev_cost, upticks, epoch = np.inf, 0, 0

        # Actual training
        while upticks < 3 or epoch < n_epochs:
            print_bias_etc(da, file_name, epoch, self.side, save_raster)

            da.set_curr_learning_rate(epoch)
            c = []
            for batch_index in range(self.n_batches):
                c.append(train_da(batch_index))
            mean_costs = np.mean(c, axis=0)
            print('Epoch: {1:3d}\n  Upticks: {2}\n  '
                  'Costs\tRecons:{0[0]:7.3f}\tL1:{0[1]:7.3f}\tL2:{0[2]:7.3f}\t'
                  'Total:{0[3]:7.3f}'
                  ''.format(mean_costs, epoch, upticks))
            if mean_costs[-1] > prev_cost:
                upticks += 1
            prev_cost = mean_costs[-1]
            epoch += 1

        print_bias_etc(da, file_name, epoch, self.side, save_raster, freq=1)

        w = da.w.get()
        norms = np.apply_along_axis(la.norm, 0, w)
        print('\nFinished training, Pruned', sum(norms <= NORM_THRESHOLD), 'nodes')
        b = da.b.get()

        return w[:, norms > NORM_THRESHOLD], b[norms > .1] if b.ndim > 0 else b

    def train_first_add_second(self, file_name):
        self.w1, self.b1 = self.train_this(self.first, self.first_train,
                                           self.n_epochs[0], file_name,
                                           save_raster=True, )

        self.x2 = theano.shared(
            np.asarray(sigmoid(self.x1.get().dot(self.w1) + self.b1), floatX),
            borrow=True)

        self.second, self.second_train = self.init_next(inpt=self.x2, index=1)

    def train_second_add_third(self):
        self.w2, self.b2 = self.train_this(self.second, self.second_train,
                                           self.n_epochs[1])

        self.x3 = theano.shared(
            np.asarray(sigmoid(self.x2.get().dot(self.w2) + self.b2), floatX),
            borrow=True)

        self.third, self.third_train = self.init_next(inpt=self.x3, index=2)

    def train_third(self, ):
        self.w3, self.b3 = self.train_this(
            self.third, self.third_train, self.n_epochs[2])

    def do_all(self, file_name):
        self.train_first_add_second(file_name)
        self.train_second_add_third()
        self.train_third()
        print(self.w1.shape, self.w2.shape, self.w3.shape)

    def save_wbs_raster(self, file_name):
        pkl_name = file_name[:-8] + '.pkl'
        with open(pkl_name, 'wb') as f:
            pickle.dump((self.w1, self.b1,
                         self.w2, self.b2,
                         self.w3, self.b3,), f)
        print('Saved pickle file', pkl_name)

        print_tile(self.w1.dot(self.w2).T,
                   file_name.format('W2'),
                   (self.side, self.side))

        print_tile(self.w1.dot(self.w2).dot(self.w3).T,
                   file_name.format('W3'),
                   (self.side, self.side))


def print_tile(data, file_name, img_shape):
    n_images = data.shape[0]
    data = data.reshape((n_images,)+img_shape)
    image = PIL.Image.fromarray(tile_raster_images(images=data, zoom=2))
    image.save(file_name)
    print("Raster saved to ", file_name)

