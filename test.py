import time
startt = time.time()
from read_fromfile import *
from libs_for_train import *
from load import mnist

train_img, test_img, train_res, test_res = mnist(onehot=True)

SPEED = 0.05
TRAIN_CIRCLES = 130
BATCHES = 128

X = T.fmatrix('x')
Y = T.fmatrix('y')

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# for i in range(100):
#     for start, end in zip(range(0, len(train_img), 128), range(128, len(train_img), 128)):
#         cost = train(train_img[start:end], train_res[start:end])
#     print np.mean(np.argmax(test_res, axis=1) == predict(test_img))

for x in range(TRAIN_CIRCLES):
	for start, end in zip(range(0, len(train_img), BATCHES), range(BATCHES, len(train_img), BATCHES)):
		cost = train(train_img[start:end], train_res[start:end])
	if x % 10 == 0:
		print np.mean(np.argmax(test_res, axis=1) == predict(test_img))
		print np.mean(np.argmax(train_res, axis=1) == predict(train_img))
		# print test_accuracy(TEST_SIZE, predict, test_img, test_res)
		# print test_accuracy(TRAIN_SIZE, predict, train_img, train_res)

print np.mean(np.argmax(test_res, axis=1) == predict(test_img))
print np.mean(np.argmax(train_res, axis=1) == predict(train_img))
# print test_accuracy(TEST_SIZE, predict, test_img, test_res)
# print test_accuracy(TRAIN_SIZE, predict, train_img, train_res)

endt = time.time()
print endt - startt
