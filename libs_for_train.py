import numpy as np
import theano
import theano.tensor as T
rng = np.random

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)
def init_weight(share):
	return theano.shared(floatX(rng.randn(*share) * 0.01))
def model(x, w_h, w_o):
	y = T.nnet.sigmoid(T.dot(x, w_h))
	return T.nnet.softmax(T.dot(y, w_o))

def test_accuracy(size, predict, test_img, test_res):
	res = predict(test_img);
	sum_res = 0 
	# print res[0], test_res[0].index(1)
	for i in range(size):
		sum_res += (res[i] == test_res[i].index(1))
	return float(sum_res) / size

def sgd(cost, params, speed=0.05):
	y = T.grad(cost=cost, wrt=params)
	update = []
	for g, p in zip(y, params):
		update.append([p, p - g * speed])
	return update