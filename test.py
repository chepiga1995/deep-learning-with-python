import time
startt = time.time()
from load import *

import theano
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

TIMES = 100
SHORTCARD = 512
trX, teX, trY, teY = TRAIN_MEL, TEST_MEL, TRAIN_RES, TEST_RES

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_h3, w_h4, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = T.nnet.sigmoid(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = T.nnet.sigmoid(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    h3 = T.nnet.sigmoid(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    h4 = T.nnet.sigmoid(T.dot(h3, w_h4))

    h4 = dropout(h4, p_drop_hidden)
    py_x = softmax(T.dot(h, w_o))
    return h, h2, h3, h4, py_x


X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((117, 1000))
w_h2 = init_weights((1000, 1000))
w_h3 = init_weights((1000, 1000))
w_h4 = init_weights((1000, 1000))
w_o = init_weights((1000, 53*53))

noise_h, noise_h2, noise_h3, noise_h4, noise_py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, 0.2, 0.5)
h, h2, h3, h4, py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_h3, w_h4, w_o]
updates = RMSprop(cost, params, lr=0.003)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(TIMES):
	for start, end in zip(range(0, len(trX), SHORTCARD), range(SHORTCARD, len(trX), SHORTCARD)):
		cost = train(trX[start:end], trY[start:end])
	if (i % 3 == 0):
		print np.mean(np.argmax(teY, axis=1) == predict(teX))
		print np.mean(np.argmax(trY, axis=1) == predict(trX))

endt = time.time()
print endt - startt