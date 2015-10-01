import numpy as np
import theano as th
import theano.tensor as tt

seed = 1000 + np.random.randint(0, 8999)
np_rng = np.random.RandomState(seed)
th_rng = tt.shared_randomstreams.RandomStreams(np_rng.randint(2**30))
floatX = th.config.floatX


class DAE():
    def __init__(self,
                 n_visible,
                 n_hidden,
                 w=None,
                 lambda1=.0,
                 lambda2=.0,
                 use_common_bias=True,
                 learn_bias=False, ):
        print("Initializing the Auto-encoder")
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if w is None:
            w = np.asarray(
                np_rng.uniform(low=-4 * (6. / (n_hidden + n_visible)) ** .5,
                               high=4 * (6. / (n_hidden + n_visible)) ** .5,
                               size=(n_visible, n_hidden)), dtype=floatX)
        self.w = th.shared(value=w, name='W', borrow=True)

        if use_common_bias:
            self.b_prime = th.shared(np.cast[floatX](-1.9))
            self.b = th.shared(np.cast[floatX](0.0))
        else:
            self.b_prime = th.shared(
                value=np.zeros(n_visible, dtype=floatX),
                name="b'", borrow=True)
            self.b = th.shared(
                value=np.zeros(n_hidden, dtype=floatX),
                name='b', borrow=True)

        self.W_prime = self.w.T
        self.x = tt.matrix('x')
        self.params = [self.w, ]
        if learn_bias:
            self.params += [self.b, self.b_prime, ]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.info = "lambda1={} lambda2={} n_visible={} n_hidden={}".format(
            lambda1, lambda2, n_visible, n_hidden)

    def get_cost_updates(self, noise, init_learning_rate, binary_data,
                         max_filter_norm=0):
        self.init_learning_rate = init_learning_rate
        self.curr_learning_rate = th.shared(
            np.cast[th.config.floatX](init_learning_rate))

        if noise == 0:
            noised = self.x

        elif binary_data:
            mask = th_rng.binomial(n=1, p=noise, size=self.x.shape)
            noised = tt.cast(tt.cast(self.x, 'int32') ^ mask, floatX)
        else:
            noised = self.x * th_rng.binomial(n=1, p=1 - noise,
                                              size=self.x.shape, dtype=floatX)

        encoded = tt.nnet.sigmoid(tt.dot(noised, self.w) + self.b)
        reconstructed = tt.nnet.sigmoid(
            tt.dot(encoded, self.W_prime) + self.b_prime)

        # Cost/Loss
        if binary_data:
            # Negative Log-likelihood
            loss = -tt.sum(self.x * tt.log(reconstructed) +
                           (1 - self.x) * tt.log(1 - reconstructed), axis=1)
        else:
            # MSE
            loss = tt.sum((self.x - reconstructed) ** 2, axis=1)

        reconstruction_loss = tt.mean(loss)
        l1_loss = tt.mean(tt.sum(abs(self.w), axis=0))
        l2_loss = tt.mean(tt.sum((self.w ** 2), axis=0))
        final_cost = reconstruction_loss + self.lambda1 * l1_loss + self.lambda2 * \
                                                              l2_loss
        costs = [reconstruction_loss, l1_loss, l2_loss, final_cost]

        gradients = tt.grad(final_cost, self.params)
        updates = []

        for param, gradient in zip(self.params, gradients):
            updated_param = param - self.curr_learning_rate * gradient

            # Weight renormalization
            if max_filter_norm and param.get_value(borrow=True).ndim == 2:
                squared_norms = tt.sum(updated_param ** 2, axis=0
                                       ).reshape((1, updated_param.shape[1]))
                scale = tt.clip(max_filter_norm / tt.sqrt(squared_norms), 0, 1)
                updated_param = updated_param * scale

            updates.append((param, updated_param))

        return costs, updates

    def set_curr_learning_rate(self, epoch):
        self.curr_learning_rate.set_value(self.init_learning_rate / (1 + epoch))

    def __str__(self):
        return 'Denoising Auto-encoder : ' + self.info