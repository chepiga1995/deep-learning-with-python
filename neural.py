import ujson
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



ARR_SHAPE = [\
    (117, 800),\
    (800, 800),\
    (800, 800),\
    (800, 800),\
    (800, 53)\
]
TIMES = 100
SHORTCARD = 256
SPEED = 0.01
DROP_INPUT = 0.2
DROP_HIDDEN = 0.6
PRINT_FROM = 10
PRINT_TO = 50
STEP_SHOW = 3
RHO = 0.9
EPSILON = 1e-4

srng = RandomStreams()
class Neural:
    def __init__(self, name="Test neural"):
        self.name = name
    def train(self, trX, teX, trY, teY, plot=True, epochs=TIMES, shortcard=SHORTCARD, speed=SPEED, drop_input=DROP_INPUT, drop_hidden=DROP_HIDDEN, step_show=STEP_SHOW, rho=RHO, epsilon=EPSILON):
        X = T.fmatrix()
        Y = T.fmatrix()
        train_set_n = len(trY)
        test_set_n = len(teY)
        accuracy_arr = []
        diff_arr = []
        i_arr = []

        noise_py_x = self._model(X, drop_input, drop_hidden)
        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
        updates = self._RMSprop(cost, lr=speed, rho=rho, epsilon=epsilon)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

        for i in range(TIMES):
            for start, end in zip(range(0, train_set_n, shortcard), range(shortcard, train_set_n, shortcard)):
                cost = train(trX[start:end], trY[start:end])
            if i % step_show == 0:
                acc = np.mean(np.argmax(teY, axis=1) == self.predict(teX))
                accuracy_arr.append(acc)
                di = self.get_diff(teX, teY)
                diff_arr.append(di)
                i_arr.append(i)
                print "{0} {1:.3f}% {2:.1f}".format(i, acc * 100, di)
        if plot:
            self._name = "Epochs: {0}, Shortcard: {1}, Speed: {2:.5f}\n Structure: {3}\n Train: {4}, Test: {5}".format(epochs, shortcard, speed, self._struct, train_set_n, test_set_n)
            self._name_f = "epochs_{0}_shortcard_{1}_speed_{2:.5f}_structure_{3}_train_{4}_test_{5}".format(epochs, shortcard, speed, self._struct, train_set_n, test_set_n)
            self._plot(i_arr, accuracy_arr, diff_arr)       
    def _floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)
    def _init_weights(self, shape):
        return theano.shared(self._floatX(np.random.randn(*shape) * 0.05))
    def _softmax(self, X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')    
    def _RMSprop(self, cost, lr=0.001, rho=0.9, epsilon=1e-4):
        grads = T.grad(cost=cost, wrt=self.arr_w)
        updates = []
        for p, g in zip(self.arr_w, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates
    def _dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X    
    def _model(self, X, p_drop_input, p_drop_hidden):
        X = self._dropout(X, p_drop_input)
        h = X
        for w in self.arr_w[:-1]:
            h = T.nnet.sigmoid(T.dot(h, w))
            h = self._dropout(h, p_drop_hidden)
        py_x = self._softmax(T.dot(h, self.arr_w[-1]))
        return py_x  
    def init_model(self, arr=ARR_SHAPE):
        self.arr_w = []
        self._struct = arr
        for shape in arr:
            self.arr_w.append(self._init_weights(shape))
        self._init_prediction()    
        return self.arr_w    
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            res = []
            for x in self.arr_w:
                res.append(x.get_value().tolist())
            ujson.dump(res, f)   
    def get_diff(self, teX, teY):
        diff = np.square(teY - self.result(teX))
        dif = np.sum(diff.tolist(), axis=1)
        di = np.sum(dif.tolist(), axis=0)
        return di  
    def print_res(self, teX, teY, map_res, start=PRINT_FROM, end=PRINT_TO):
        for i, x in enumerate(self.result(teX[start:end])):
            for y in x:
                if y > 0.05:
                    print "{0:} {1:.1f}  ".format(map_res[x.tolist().index(y)], y * 100),
            print map_res[np.argmax(teY[i + start], axis=0)]   
    def _plot(self, i_arr, accuracy_arr, diff_arr):
        plt.plot(i_arr, accuracy_arr, 'b')
        plt.ylabel('Blue - accuracy')
        plt.axis([min(i_arr), max(i_arr), min(accuracy_arr), max(accuracy_arr)])
        plt.xlabel('Test Accuracy ' + self._name)
        plt.grid(True)
        plt.savefig('plots/accuracy_' + self._name_f + '.png')
        plt.plot(i_arr, diff_arr, 'b')
        plt.ylabel('Blue - diff')
        plt.axis([min(i_arr), max(i_arr), min(diff_arr), max(diff_arr)])
        plt.xlabel('Test diff ' + self._name)
        plt.grid(True)
        plt.savefig('plots/diff_' + self._name_f + '.png')
    def _init_prediction(self):
        X = T.fmatrix()
        py_x = self._model(X, 0., 0.)
        y_x = T.argmax(py_x, axis=1)
        self.predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True) 
        self.result = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)
    def load_from_file(self, filename):
        arr_w = []
        with open(filename) as data_file:  
            temp = ujson.load(data_file)
            for w in temp:
                arr_w.append(theano.shared(np.asarray(w, dtype=theano.config.floatX)))
        self.arr_w = arr_w    
        self._init_prediction()    