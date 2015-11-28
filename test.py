import time
start = time.time()
from read_fromfile import *
from libs_for_train import *

SPEED = 0.05
TRAIN_CIRCLES = 150
BATCHES = 150


w_h = init_weight((SIZE * SIZE, 3000))
w_o = init_weight((3000, 10))
X = T.fmatrix('x')
Y = T.fmatrix('y')

py_x = model(X, w_h, w_o)
pred_y = T.argmax(py_x, axis=1)
# print theano.function([X], py_x)(result[0])

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
update = sgd(cost, [w_h, w_o], SPEED)

train = theano.function(inputs=[X,Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=pred_y, allow_input_downcast=True)

for x in range(TRAIN_CIRCLES):
	for start, end in zip(range(0, len(test_img), BATCHES), range(BATCHES, len(test_img), BATCHES)):
		train(train_img[start:end], train_res[start:end])
	if x % 10 == 0:
		print test_accuracy(TEST_SIZE, predict, test_img, test_res)
		print test_accuracy(TRAIN_SIZE, predict, train_img, train_res)

print test_accuracy(TEST_SIZE, predict, test_img, test_res)
print test_accuracy(TRAIN_SIZE, predict, train_img, train_res)

end = time.time()
print end - start
# D = (result, labels)
# y = model()
# x = T.dmatrix('x')

# print w
# N = 4
# feats = 5
# D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
# x = T.dmatrix('x')
# s = T.sum(1 / (1 + T.exp(-x)))
# gs = T.grad(s, x)
# print D[0], theano.function([x], gs)(D[0])



# training_steps = 10000
# print "Train: ", D[0]
# print "Result: ", D[1]
# # Declare Theano symbolic variables
# x = T.matrix("x")
# y = T.vector("y")
# w = theano.shared(rng.randn(feats), name="w")
# b = theano.shared(0., name="b")
# print("Initial model:")
# print(w.get_value())
# print(b.get_value())

# # Construct Theano expression graph
# print "matrix: ", theano.function([x], T.dot(x, w) - b)(D[0])
# print "matrix: ", theano.function([x], T.dot(x, w))(D[0])
# p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
# prediction = p_1 > 0.5                    # The prediction thresholded
# xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
# cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
# gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
#                                           # (we shall return to this in a
#                                           # following section of this tutorial)

# # Compile
# train = theano.function(
#           inputs=[x,y],
#           outputs=[prediction, xent],
#           updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
# predict = theano.function(inputs=[x], outputs=prediction)

# # Train
# for i in range(training_steps):
#     pred, err = train(D[0], D[1])

# print("Final model:")
# print(w.get_value())
# print(b.get_value())
# print("target values for D:")
# print(D[1])
# print("prediction on D:")
# print(predict(D[0]))